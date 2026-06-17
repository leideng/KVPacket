import torch
import argparse
import sys
import random
import os
import gc
import re
from time import perf_counter
from alive_progress import alive_it, alive_bar
from rich.pretty import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from kv_packet.dataset import get_ret_eval_generator
from kv_packet.utils.generate import GenerationCache, TokenizerType
from kv_packet.packet_wrapper import PacketWrapper, WrapperStateDict
from kv_packet.model import SupportedModel
from kv_packet.utils.config import gather_config_files, load_config_file
from kv_packet.utils.train_filler import (
    TrainSample,
    TrainConfig,
    sample_to_str,
    load_train_config,
    prepare_sample_input,
    batched_input_embed,
    packet_4d_mask,
    batched_packet_4d_mask,
    get_packed_labels,
    get_packed_logits,
    build_generation_cache,
    dtype_map
)

def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def print_duration(label: str, seconds: float) -> None:
    print(f"[Timing] {label}: {seconds:.3f} s ({format_duration(seconds)})")


def print_sample_length_statistics(
    samples: list[TrainSample],
    tokenizer: TokenizerType,
    header_len: int,
    trailer_len: int,
) -> None:
    # [Training-length stats change] Report real token lengths after sampling/templates.
    """Print token-length distributions for the formatted training samples."""
    if not samples:
        return

    def token_len(text: str) -> int:
        # Match training tokenization: templates already exist in each sample,
        # and model-specific special tokens must not be added a second time.
        tokens = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return len(tokens["input_ids"])

    def print_summary(label: str, values: list[int]) -> None:
        values = sorted(values)

        def percentile(ratio: float) -> int:
            return values[round((len(values) - 1) * ratio)]

        # P90：90% 的样本长度不超过该值，剩余 10% 更长
        # P95：95% 的样本长度不超过该值，剩余 5% 更长
        print(
            f"  {label}: mean={sum(values) / len(values):.2f}, "
            f"median={percentile(0.5)}, p90={percentile(0.9)}, "
            f"p95={percentile(0.95)}, min={values[0]}, max={values[-1]}"
        )
    
    # 每条样本完整输入的 token 数
    # 对应 teacher logits 生成以及 full-recompute 使用的输入
    full_input_lengths: list[int] = []
    # 每条样本中所有 document 分别 tokenize 后的 token 数之和 
    # len(doc1) + len(doc2) + ... + len(docN)
    document_totals: list[int] = [] 
    document_lengths: list[int] = []
    # 每条样本经过 KVPacket 包装后的输入上下文长度
    # preamble长度 + 所有document长度 + task_prompt长度 + document数量 × (header_len + trailer_len)
    packet_context_lengths: list[int] = []
    # document_counts[i] 表示第 i 条训练样本包含的 document 数量
    document_counts: list[int] = []

    for sample in samples:
        # KVPacket tokenizes every document independently before wrapping it.
        sample_document_lengths = [
            token_len(document) for document in sample["documents"]
        ]
        document_total = sum(sample_document_lengths)
        document_count = len(sample_document_lengths)

        # This is the exact concatenated prompt used to generate teacher logits
        # and is also the relevant input length for full-recompute.
        full_input_lengths.append(token_len(sample_to_str(sample)))
        document_totals.append(document_total)
        document_lengths.extend(sample_document_lengths)
        document_counts.append(document_count)

        # Each document receives its own learned header and trailer tokens.
        # Teacher-generated answer tokens are intentionally excluded here.
        packet_context_lengths.append(
            token_len(sample["preamble"])
            + document_total
            + token_len(sample["task_prompt"])
            + document_count * (header_len + trailer_len)
        )

    print("*" * 50)
    print("Training input length statistics (tokens):")
    print_summary("full input per sample", full_input_lengths)
    print("-" * 50)
    print_summary("all documents per sample", document_totals)
    print("-" * 50)
    if document_lengths:
        print_summary("individual document", document_lengths)
    print("-" * 50)
    print_summary("document count per sample", document_counts)
    print("-" * 50)
    print_summary("KVPacket context per sample", packet_context_lengths)
    print("*" * 50)


def train_wrapper_4d_batch(
    samples: list[TrainSample],
    model: SupportedModel,
    tokenizer: TokenizerType,
    packet_wrapper: PacketWrapper,
    batch_size: int,
    gen_batch_size: int,
    use_logits: bool,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
    generation_cache: GenerationCache|None=None,
    generation_config: GenerationConfig|None=None,
    device: torch.device = torch.device("cuda:0"),
    forward_batch_size: int = -1,
    epoch: int = -1,
    epoch_indices: list[int]|None = None,
):
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    if forward_batch_size <= 0:
        forward_batch_size = batch_size
    
    if batch_size % forward_batch_size != 0:
        raise ValueError("batch_size must be multiple of forward_batch_size")

    if len(samples) % batch_size != 0:
        raise ValueError("Number of samples must be multiple of batch_size")

    # No-op if main pre-filled cache; generates missing samples when called standalone.
    generation_cache = build_generation_cache(
        samples,
        gen_batch_size,
        model,
        tokenizer,
        generation_config,
        generation_cache,
        store_logits=use_logits,
    )

    # Training loop
    train_step = 0
    eval_tokens = 0
    acc_loss = 0.0

    # Shuffle samples
    num_samples = len(samples)
    # rand_indices = list(range(num_samples))
    # random.shuffle(rand_indices)
    if epoch_indices is None:
        epoch_indices = list(range(num_samples))
    else:
        assert len(epoch_indices) == num_samples

    batched_indices_list: list[list[int]] = []
    for i in range(0, num_samples, forward_batch_size):
        batched_indices_list.append(epoch_indices[i: i + forward_batch_size])

    samples_bar = alive_it(batched_indices_list)
    samples_bar.title = f"Train epoch {epoch}" if epoch >= 0 else "Train" # type: ignore

    if hasattr(model.config, "sliding_window"):
        sliding_window = model.config.sliding_window
        assert isinstance(sliding_window, int|None)
    else:
        sliding_window = None

    with alive_bar(total=num_samples, title="Training") as bar:
        for batched_indices in batched_indices_list:
            batched_samples = [samples[idx] for idx in batched_indices]
            samples_bar.text = f"Train step {train_step}/{len(samples)}" # type: ignore
            batch_input_embeds: list[torch.Tensor] = []
            batch_input_chunk_sizes: list[list[int]] = []
            batch_query_lens: list[int] = []
            batch_gen_seq_lens: list[int] = []

            for sample in batched_samples:
                prepared_input = prepare_sample_input(
                    sample,
                    model,
                    generation_cache,
                    tokenizer,
                    packet_wrapper,
                    device,
                )
                batch_input_embeds.append(prepared_input[0])
                batch_input_chunk_sizes.append(prepared_input[1])
                batch_query_lens.append(prepared_input[2])
                batch_gen_seq_lens.append(prepared_input[3])

            input_embed = batched_input_embed(batch_input_embeds)
            packet_attn_mask = batched_packet_4d_mask(
                batch_input_chunk_sizes,
                batch_query_lens,
                sliding_window=sliding_window,
                device=device,
            )

            max_gen_seq_len = max(batch_gen_seq_lens)
            eval_mask = torch.zeros(
                (len(batched_samples), max_gen_seq_len),
                dtype=input_embed.dtype,
                device=device
            ) # [f_batch_size, max_gen_seq_len]

            for i, gen_seq_len in enumerate(batch_gen_seq_lens):
                eval_mask[i, -gen_seq_len:] = 1.0

            # with packet_attn_mask, we will not do full attention;
            # only query tokens can attend to all input chunks;
            # one chunk does not attend to other chunks
            # (NOTE) Here we do not reuse precomputed KV caches; 
            # instead we use sparse attention with given mask
            # to minic the KV cache reuse behavior;
            outputs = model(
                inputs_embeds=input_embed,
                attention_mask=packet_attn_mask,
            )

            logits = outputs.logits
            assert isinstance(logits, torch.Tensor) # [f_batch_size, seq_len, vocab_size]

            logits_to_eval = logits[:, -max_gen_seq_len:, :]
            num_tokens = int(eval_mask.sum().item())
            eval_tokens += num_tokens

            if use_logits:
                # [KVPacket disk-cache change] Disk-loaded logits are moved to
                # the training device before KL loss.
                target_logits = get_packed_logits(
                    batched_samples,
                    generation_cache,
                    padding_side='left',
                ) # [f_batch_size, max_gen_seq_len, vocab_size]
                target_logits = target_logits.log_softmax(dim=-1)
                logits_to_eval = logits_to_eval.log_softmax(dim=-1)

                assert target_logits.size() == logits_to_eval.size()

                loss_fct: torch.nn.Module = torch.nn.KLDivLoss(
                    reduction="none",
                    log_target=True  # input log Q instead of Q
                )

                # KV (target || input) =KV(P || Q) 
                loss = loss_fct(
                    input=logits_to_eval, #log Q (carry gradients)
                    target=target_logits, #log P
                ).sum(-1) # [f_batch_size, max_gen_seq_len]

                loss = (loss * eval_mask).sum()
            
            else:
                # [KVPacket disk-cache change] Keep label path consistent with
                # CPU/disk cache loading.
                target_ids = get_packed_labels(
                    batched_samples,
                    generation_cache,
                    padding_side='left',
                ).reshape(-1) # [f_batch_size * max_gen_seq_len]

                loss_fct = torch.nn.CrossEntropyLoss(
                    reduction="none"
                )

                loss = loss_fct(
                    logits_to_eval.reshape(-1, logits_to_eval.size(-1)),
                    target_ids,
                ) # [f_batch_size * max_gen_seq_len]

                loss = (loss * eval_mask.reshape(-1)).sum()

            # Gradient accumulation: d/dθ(Σ L_i) = Σ dL_i/dθ; backward per forward_batch_size
            # matches one backward on the summed loss (grads accumulate in .grad).
            loss.backward()
            acc_loss += loss.item()  # logging only not for backward pass (gradient accumulation)

            train_step += forward_batch_size
            bar(forward_batch_size)

            if train_step % batch_size == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                lr = scheduler.get_last_lr()[0]
                print(f"eval tokens {eval_tokens}, loss {acc_loss / eval_tokens:.4f}, lr {lr:.3e}")
                eval_tokens = 0
                acc_loss = 0.0


def train_wrapper_4d(
    samples: list[TrainSample],
    model: SupportedModel,
    tokenizer: TokenizerType,
    packet_wrapper: PacketWrapper,
    batch_size: int,
    gen_batch_size: int,
    use_logits: bool,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimizer: torch.optim.Optimizer,
    generation_cache: GenerationCache|None=None,
    generation_config: GenerationConfig|None=None,
    # cache_path: str|None=None,
    device: torch.device = torch.device("cuda:0"),
    epoch: int = -1,
    epoch_indices: list[int]|None = None,
):
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.gradient_checkpointing_enable()

    # 为当前 sample 生成或者补全缓存
    # 这一步会把每个 sample 的生成结果（以及 logits，如果 use_logits=True）存入 generation_cache
    # 这样后面训练循环中就可以直接从 cache 取出生成结果而不需要重复生成 model.generate
    generation_cache = build_generation_cache(
        samples,
        gen_batch_size,
        model,
        tokenizer,
        generation_config,
        generation_cache,
        store_logits=use_logits,
    )

    # Training loop
    train_step = 0
    eval_tokens = 0
    acc_loss = 0.0
    # Shuffle samples
    # rand_indices = list(range(len(samples)))
    # random.shuffle(rand_indices)

    if epoch_indices is None:
        epoch_indices = list(range(len(samples)))
    else:
        assert len(epoch_indices) == len(samples)

    samples_bar = alive_it(epoch_indices)
    samples_bar.title = f"Train epoch {epoch}" if epoch >= 0 else "Train" # type: ignore

    # 逐个 sample 训练，“单样本 forward” 模式
    for idx in samples_bar:
        sample = samples[idx]
        samples_bar.text = f"Train step {train_step}/{len(samples)}" # type: ignore
        gen = generation_cache.get(sample_to_str(sample), device=device) # 从 cache 里取当前 sample 的生成结果 (sequences/logits)
        assert gen is not None
        assert len(gen["sequences"]) == 1
        assert len(gen["logits"]) == 1 or not use_logits

        # 记录各段输入的长度与 embedding (preamble + documents + query)，后续拼接成一个 batch 输入模型
        input_chunk_sizes: list[int] = [] 
        input_embed_list: list[torch.Tensor] = []

        if sample["preamble"]:
            # 如果 sample 有 preamble few-shot 前缀
            context_input = tokenizer(
                [sample["preamble"]],
                add_special_tokens=False,
                return_tensors="pt",
            )
            context_ids = context_input["input_ids"]
            assert isinstance(context_ids, torch.Tensor)

            # 记录这段长度
            input_chunk_sizes.append(context_ids.size(1))

            # 计算 embedding 并记录
            with torch.no_grad():
                context_embed = model.model.embed_tokens(context_ids.to(device))
            input_embed_list.append(context_embed)

        # 记录各个文档的长度和 embedding
        for data_str in sample["documents"]:
            data_input = tokenizer(
                [data_str],
                add_special_tokens=False,
                return_tensors="pt",
            )
            data_ids = data_input["input_ids"]
            assert isinstance(data_ids, torch.Tensor)

            with torch.no_grad():
                data_embed = model.model.embed_tokens(data_ids.to(device))
            wrapped_data_embed = packet_wrapper.wrap(data_embed)

            input_chunk_sizes.append(wrapped_data_embed.size(1))
            input_embed_list.append(wrapped_data_embed)

        query_input = tokenizer(
            sample["task_prompt"],
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        query_ids = query_input["input_ids"]
        assert isinstance(query_ids, torch.Tensor)

        # 取当前 sample 的目标生成序列
        gen_seq = gen["sequences"][0].to(device)

        # We only need to predict up to the last token, thus the input is gen_seq[:-1]
        # 拼接 query_ids 和 gen_seq[:-1] 作为模型输入
        # 模型要预测目标序列的下一个 token，所以输入里不需要最后一个 token
        query_ids = torch.cat([query_ids, gen_seq[:-1].unsqueeze(0)], dim=1)

        # 记录 query + gen_seq 的长度（即模型输入的长度）
        query_len = query_ids.size(1)
        gen_seq_len = gen_seq.size(0)

        if hasattr(model.config, "sliding_window"):
            sliding_window = model.config.sliding_window
            assert isinstance(sliding_window, int|None)
        else:
            sliding_window = None
        
        # 生成 4D attention mask，使 query token 能够看见前面 chunk
        # 但是文档 chunk 之间只自注意力，不互相交互
        packet_attn_mask = packet_4d_mask(
            input_chunk_sizes=input_chunk_sizes,
            query_len=query_len,
            sliding_window=sliding_window,
            device=device,
        )

        # 计算 query token embedding，并加入 input_embed_list
        with torch.no_grad():
            query_embed = model.model.embed_tokens(query_ids)
        input_embed_list.append(query_embed)

        # 将所有 chunk embedding 和 query embedding 拼接成一个大输入
        input_embed = torch.cat(input_embed_list, dim=1)

        # 前向计算模型 logits
        outputs = model(
            inputs_embeds=input_embed,
            attention_mask=packet_attn_mask,
        )
        logits = outputs.logits

        # gen_seq 是为该 sample 预先生成并缓存的参考生成序列 (token id)
        # 训练任务是让模型重现这个序列或者它的 logits，所以 gen_seq 是目标序列
        # 最后连续的 gen_seq_len 个 token 是需要评估的生成部分，计算 loss 时只评估这部分
        assert isinstance(logits, torch.Tensor)
        logits_to_eval = logits[:, -gen_seq_len:, :]

        # 统计本次训练消耗的 eval token 数量
        num_tokens = logits_to_eval.size(1)
        eval_tokens += num_tokens

        if use_logits:
            loss_fct: torch.nn.Module = torch.nn.KLDivLoss(
                reduction="sum",
                log_target=True
            )
            # 从 generation_cache 取目标 logits
            gen_logits = gen["logits"][0].to(device)
            # 对预测 logtis 和 目标 logits 都做 log_softmax
            logits_to_eval = logits_to_eval.log_softmax(dim=-1)
            target_logits = torch.nn.functional.log_softmax(gen_logits, dim=-1)
            loss = loss_fct(
                input=logits_to_eval.reshape(-1, logits_to_eval.size(-1)),
                target=target_logits.reshape(-1, logits_to_eval.size(-1)),
            )
        else:
            # 直接用目标 token ids 计算交叉熵 loss
            target_ids = gen_seq.unsqueeze(0)
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction="sum"
            )
            loss = loss_fct(
                input=logits_to_eval.reshape(-1, logits_to_eval.size(-1)),
                target=target_ids.reshape(-1),
            )

        # 释放中间张量
        del outputs, logits, input_embed, packet_attn_mask
        # 清理未使用的显存
        torch.cuda.empty_cache()
        # 反向传播梯度，只有 wrapper 的 header/trailer 参数会更新
        loss.backward()
        acc_loss += loss.detach().item()
        train_step += 1

        # 当已经处理了一个 batch 的样本后
        # 进行一次优化器更新和学习率调度
        if train_step % batch_size == 0:
            # 执行优化器更新
            optimizer.step()
            # 清零梯度
            optimizer.zero_grad()
            # 学习率调度器更新
            scheduler.step()
            # 取当前学习率
            lr = scheduler.get_last_lr()[0]
            print(f"eval tokens {eval_tokens}, loss {acc_loss / eval_tokens:.4f}, lr {lr:.3e}")
            # 重置统计，准备下一个 batch
            eval_tokens = 0
            acc_loss = 0.0
            # 减少内存碎片
            gc.collect()
            torch.cuda.empty_cache()


    if train_step % batch_size != 0:
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()


class TrainCache:
    def __init__(self):
        self.tokenizer_cache: dict[str, TokenizerType] = {}
        self.model_cache: dict[tuple[str, str, str], SupportedModel] = {}


def train_one_config(
    train_config: TrainConfig,
    train_cache: TrainCache,
):
    print("Training configuration:")
    pprint(train_config)

    if os.path.exists(train_config["save_path"]) is False:
        os.makedirs(train_config["save_path"], exist_ok=True)

    # 是否使用进程内缓存来重用 tokenizer / model
    use_cache = train_config["use_cache"]

    # 根据配置确定模型加载的设备映射（支持 "auto"）
    if train_config["model"]["device"] == "auto":
        device_map: torch.device|str= "auto"
        model_device = torch.device("cuda:0")
    else:
        device_map = torch.device(train_config["model"]["device"])
        model_device = torch.device(train_config["model"]["device"])

    if model_device.type == "cuda":
        torch.cuda.set_device(model_device)

    print(
        f"Model device: {model_device}; "
        f"current CUDA device: "
        f"{torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}"
    )

    # 加载或从缓存取 tokenizer（并确保有 pad token，padding 方向为左）
    if not use_cache or train_cache.tokenizer_cache.get(train_config["model"]["model_path"]) is None:
        tokenizer: TokenizerType = AutoTokenizer.from_pretrained(
            train_config["model"]["model_path"],
        )
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            # 若 tokenizer 没有 pad token，则用 eos 作为 pad
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if use_cache:
            # 将 tokenizer 放入进程缓存以便后续复用
            train_cache.tokenizer_cache[train_config["model"]["model_path"]] = tokenizer
    else:
        tokenizer = train_cache.tokenizer_cache[train_config["model"]["model_path"]]

    model_key: tuple[str, str, str] = (
        train_config["model"]["model_path"],
        train_config["model"]["dtype"],
        train_config["model"]["device"],
    )

    # 加载或复用模型（Cached）
    # 设置 model 的 generation_config.pad_token_id 并置为 eval
    if train_cache.model_cache.get(model_key) is None:
        loaded_model = AutoModelForCausalLM.from_pretrained(
            train_config["model"]["model_path"],
            dtype=train_config["model"]["dtype"],
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        assert isinstance(loaded_model, SupportedModel)
        model = loaded_model
        assert model.generation_config is not None
        # 保证 generation 时使用与 tokenizer 一致的 pad token id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        train_cache.model_cache[model_key] = model
    else:
        model = train_cache.model_cache[model_key]

    # # cache 使用的设备（用于 GenerationCache 存放/载入）
    # cache_device = torch.device(train_config["cache_device"])
    cache_device_config = train_config["cache_device"]
    # [KVPacket disk-cache change] Treat cache_device="disk" as CPU metadata +
    # per-sample logits shards under "<cache_path>.shards".
    if cache_device_config == "disk":
        cache_device = torch.device("cpu")
        assert train_config["cache_path"] is not None, (
            "cache_path must be set when cache_device is 'disk'."
        )
        cache_offload_dir = f"{train_config['cache_path']}.shards"
    else:
        cache_device = torch.device(cache_device_config)
        cache_offload_dir = None
    
    # 计算 embedding 的 mean/std（用于初始化 PacketWrapper 的分布参数）
    mean = torch.mean(model.model.embed_tokens.weight).item()
    std = torch.std(model.model.embed_tokens.weight).item()
    print(f"Embedding mean: {mean:.3e}, std: {std:.3e}")

    # wrapper 使用的 dtype：优先使用 train_config['dtype']，否则使用 model dtype
    wrapper_dtype = train_config["dtype"] if train_config["dtype"] is not None else train_config["model"]["dtype"]

    # 控制是否从已有 checkpoint 继续训练，true 启用恢复训练，false 从头训练
    resume = train_config["resume"] 
    # 指定恢复哪一个 epoch 的 checkpoint，如果为 None 则自动查找最新的 epoch checkpoint
    resume_epoch: int|None = train_config["resume_epoch"] 
    resume_checkpoint: str|None = None

    if resume:
        save_path = train_config["save_path"]
        file_name = train_config["file_name"]

        if resume_epoch is not None:
            # 指定 epoch 恢复点
            target = os.path.join(save_path, f"{file_name}.epoch{resume_epoch}")
            if not os.path.exists(target):
                raise ValueError(f"Checkpoint for epoch {resume_epoch} not found: {target}")
            resume_checkpoint = target
        else:
            # 未指定 epoch，查找目录下最新的 epoch 文件
            ckpt_files = [
                f for f in os.listdir(save_path)
                if re.search(rf"^{re.escape(file_name)}\.epoch(\d+)$", f)
            ]
            if ckpt_files:
                ckpt_files.sort(key=lambda f: int(m.group(1)) if (m := re.search(r"\.epoch(\d+)$", f)) else -1)
                resume_checkpoint = os.path.join(save_path, ckpt_files[-1])
            else:
                print(f"No checkpoints found in {save_path}, starting from scratch.")

    if resume_checkpoint is not None:
        # 如果存在 checkpoint，载入 header/trailer 并通过 PacketWrapper.from_state_dict 恢复
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=model_device)
        header = checkpoint["header"]
        trailer = checkpoint["trailer"]

        assert isinstance(header, torch.Tensor)
        assert isinstance(trailer, torch.Tensor)

        packet_wrapper = PacketWrapper.from_state_dict(
            WrapperStateDict(
                header=header,
                trailer=trailer,
                train_config=checkpoint.get("train_config", None),
            ),
            device=model_device
        )
        # 校验恢复的 wrapper 与当前 model/config 是否一致
        assert packet_wrapper.dim == model.config.hidden_size, (
            f"Packet wrapper dim {packet_wrapper.dim} does not match model hidden size {model.config.hidden_size}"
        )
        assert packet_wrapper.header_len == train_config["header_len"], (
            f"Packet wrapper header len {packet_wrapper.header_len} does not match training config {train_config['header_len']}"
        )
        assert packet_wrapper.trailer_len == train_config["trailer_len"], (
            f"Packet wrapper trailer len {packet_wrapper.trailer_len} does not match training config {train_config['trailer_len']}"
        )
        epoch_match = re.search(r"\.epoch(\d+)$", resume_checkpoint)
        assert epoch_match, "Could not parse epoch number from checkpoint filename."
        start_epoch = int(epoch_match.group(1))
        print(f"Resuming from epoch: {start_epoch}")
    else:
        # 未找到恢复点时，创建新的 PacketWrapper（header/trailer 由正态分布初始化）
        assert model.config.hidden_size is not None, "Model config must have hidden_size defined."
        packet_wrapper = PacketWrapper(
            header_len=train_config["header_len"],
            trailer_len=train_config["trailer_len"],
            dim=model.config.hidden_size,
            dtype=dtype_map[wrapper_dtype],
            mean=mean,
            std=std,
            device=model_device
        )
        start_epoch = 0

    # 优化器只优化 packet_wrapper 的 header 与 trailer（其余模型参数冻结）
    optimizer_config = train_config["opt_config"]
    optimizer = torch.optim.AdamW(
        params=[
            packet_wrapper.header,
            packet_wrapper.trailer,
        ],
        **optimizer_config
    )

    # 构建训练样本：
    # 从每个 data_config 使用 get_ret_eval_generator 生成 TrainSample 列表
    samples: list[TrainSample] = []

    # 对每个 data_config 生成对应的样本生衡器
    # 并将 RetEvalEntry 转换为 TrainSample 实例后添加到总样本列表中
    for data_config in train_config["data_configs"]:
        eval_generator = get_ret_eval_generator(
            name=data_config["dataset_name"],
            num_samples=data_config["num_samples"],
            num_data_strs=data_config["num_data_strs"],
            num_shots=data_config["num_shots"],
            subset=data_config["subset"],
            split=data_config["split"],
            seed=data_config["seed"],
            data_kwargs=data_config["data_kwargs"],
            template=data_config["template"],
            template_kwargs=data_config["template_kwargs"],
        )
        samples.extend([TrainSample(**sample,) for sample in eval_generator]) # type: ignore

    # 样本统计与批次计算（用于构造 scheduler total iters）
    num_samples = len(samples) # 总样本数：biography256, hotpotqa512
    print(f"Total training samples: {num_samples}")

    # Report lengths after dataset filtering, sampling, and chat templating so
    # the statistics describe the exact samples used by this training run.
    # [Training-length stats change] This is diagnostic only; it does not affect training.
    print_sample_length_statistics(
        samples=samples,
        tokenizer=tokenizer,
        header_len=train_config["header_len"],
        trailer_len=train_config["trailer_len"],
    )

    if num_samples % train_config["batch_size"] != 0:
        print(f"Warning: number of samples {num_samples} is not divisible by batch size {train_config['batch_size']}.")

    iter_per_epoch = num_samples // train_config["batch_size"]
    total_iter = iter_per_epoch * train_config["total_epoch"]

    # 学习率调度器：如果配置中未设置 total_iters，则补上计算值
    scheduler_config = train_config["scheduler_config"]

    if scheduler_config.get("total_iters", 0) == 0:
        scheduler_config["total_iters"] = total_iter

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        **scheduler_config
    )

    # 固定随机种子并初始化 epoch 索引列表
    random.seed(train_config["seed"])
    epoch_indices = list(range(num_samples))

    if start_epoch > 0:
        # 计算让学习率调度器 scheduler 前进的步数
        steps_to_skip = start_epoch * iter_per_epoch
        print(f"Advancing scheduler {steps_to_skip} steps...")
        for _ in range(steps_to_skip):
            # 前进 scheduler 以调整学习率保持 lr 与 epoch 对齐
            scheduler.step()
    
        for _ in range(start_epoch):
            # 重演每个历史 epoch 的样本打乱过程
            # 以保持 epoch 内样本顺序与之前训练一致
            random.shuffle(epoch_indices)

    # GenerationCache：根据 cache_path 载入或初始化
    cache_path = train_config["cache_path"]
    cache_load_start = perf_counter()
    if cache_path is not None:
        if os.path.exists(cache_path):
            generation_cache = GenerationCache.load_from_file(cache_path, device=cache_device)
            if cache_offload_dir is not None:
                # [KVPacket disk-cache change] Convert legacy/resident cache to shards.
                generation_cache.enable_offload(cache_offload_dir)
            print(f"Loaded generation cache from {cache_path}, size: {len(generation_cache.cache)}")
        else:
            # generation_cache = GenerationCache(device=cache_device)
            generation_cache = GenerationCache(
                device=cache_device,
                # [KVPacket disk-cache change] New samples are written directly to shards.
                offload_dir=cache_offload_dir,
            )
            print(f"Initialized new generation cache at {cache_path}")
    else:
        # generation_cache = GenerationCache(device=cache_device)
        generation_cache = GenerationCache(
            device=cache_device,
            # [KVPacket disk-cache change] Allows disk offload even without cache_path.
            offload_dir=cache_offload_dir,
        )
    cache_load_seconds = perf_counter() - cache_load_start
    print_duration("Teacher cache load", cache_load_seconds)

    # 若配置了 generation kwargs，则构造 GenerationConfig
    if train_config['model']["generation_kwargs"]:
        generation_config = GenerationConfig(
            **train_config["model"]["generation_kwargs"]
        )
    else:
        generation_config = None

    # 读取常用训练参数到局部变量，方便传参
    batch_size = train_config["batch_size"]
    gen_batch_size = train_config["gen_batch_size"]
    forward_batch_size = train_config["forward_batch_size"]
    total_epoch = train_config["total_epoch"]
    use_logits = train_config["use_logits"] # True 时需要缓存 logits 以计算 KL-loss，否则生成的序列作为目标 token
    cache_path = train_config["cache_path"]

    # 构建/更新 generation cache：只对缺失样本做生成并缓存
    old_cache_len = len(generation_cache.cache)

    # checkpoint
    missing_teacher_samples = sum(
        sample_to_str(sample) not in generation_cache
        for sample in samples
    )
    print(
        f"Teacher targets: {old_cache_len} cached, "
        f"{missing_teacher_samples} to generate."
    )

    synchronize_cuda(model_device)
    teacher_start = perf_counter()

    generation_cache = build_generation_cache(
        samples,
        gen_batch_size,
        model,
        tokenizer,
        generation_config,
        generation_cache,
        store_logits=use_logits,
    )
    new_cache_len = len(generation_cache.cache)

    # 若 cache 扩增并指定了 cache_path，则将更新后的 cache 存盘
    if new_cache_len > old_cache_len and cache_path is not None:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(generation_cache.to_state_dict(), cache_path)
        print(f"Saved updated generation cache to {cache_path}, size: {new_cache_len}")

    # checkpoint
    synchronize_cuda(model_device)
    teacher_seconds = perf_counter() - teacher_start
    print_duration(
        f"Teacher logits precompute ({new_cache_len - old_cache_len} samples)",
        teacher_seconds,
    )

    # checkpoing
    epoch_seconds: list[float] = []
    synchronize_cuda(model_device)
    training_start = perf_counter()

    # 进入训练主循环：按 epoch 迭代，选择两种训练函数之一
    try:
        for epoch in range(start_epoch, total_epoch):
            print(f"Epoch {epoch + 1}/{total_epoch}")
            synchronize_cuda(model_device)
            epoch_start = perf_counter()
            random.shuffle(epoch_indices) # 随机打乱样本索引
            if forward_batch_size == 1: # this is faster for batch size 1
                # 单样本 forward 路径，启用 gradient checkpointing 的实现
                train_wrapper_4d(
                    samples=samples,
                    model=model,
                    batch_size=batch_size,
                    gen_batch_size=gen_batch_size,
                    use_logits=use_logits,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokenizer=tokenizer,
                    packet_wrapper=packet_wrapper,
                    generation_cache=generation_cache,
                    generation_config=generation_config,
                    device=model_device,
                    epoch=epoch,
                    epoch_indices=epoch_indices,
                )
            else:
                # 批量子 forward 路径：把一个 batch 划分为多个子 forward 以节省显存
                train_wrapper_4d_batch(
                    samples=samples,
                    model=model,
                    batch_size=batch_size,
                    gen_batch_size=gen_batch_size,
                    use_logits=use_logits,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokenizer=tokenizer,
                    packet_wrapper=packet_wrapper,
                    generation_cache=generation_cache,
                    generation_config=generation_config,
                    device=model_device,
                    forward_batch_size=forward_batch_size,
                    epoch=epoch,
                    epoch_indices=epoch_indices,
                )
            # 每个 epoch 后清理内存，并在配置的 ckpt 周期保存中间模型
            gc.collect()
            torch.cuda.empty_cache()
            synchronize_cuda(model_device)
            current_epoch_seconds = perf_counter() - epoch_start
            epoch_seconds.append(current_epoch_seconds)
            print_duration(
                f"Training epoch {epoch + 1}/{total_epoch}",
                current_epoch_seconds,
            )
            if train_config["ckpt_epoch"] > 0 and (epoch + 1) % train_config["ckpt_epoch"] == 0 and (epoch + 1) < total_epoch:
                state_dict = packet_wrapper.state_dict()
                state_dict["train_config"] = train_config # type: ignore
                torch.save(state_dict, f"{train_config['save_path']}/{train_config['file_name']}.epoch{epoch + 1}")
                print(f"Saved checkpoint to {train_config['save_path']}/{train_config['file_name']}.epoch{epoch + 1}")

        # 训练结束后保存最终 wrapper（包括 train_config）
        state_dict = packet_wrapper.state_dict()
        state_dict["train_config"] = train_config # type: ignore
        torch.save(state_dict, f"{train_config['save_path']}/{train_config['file_name']}")
        print(f"Saved trained packet to {train_config['save_path']}/{train_config['file_name']}")

        synchronize_cuda(model_device)
        training_stage_seconds = perf_counter() - training_start
        optimization_seconds = sum(epoch_seconds)

        print("\n========== Timing Summary ==========")
        print_duration("Teacher cache load", cache_load_seconds)
        print_duration("Teacher logits precompute", teacher_seconds)
        print_duration("Wrapper optimization total", optimization_seconds)
        print_duration(
            "Training stage wall time (including checkpoint I/O)",
            training_stage_seconds,
        )
        if epoch_seconds:
            print_duration(
                "Average training time per epoch",
                sum(epoch_seconds) / len(epoch_seconds),
            )
        print_duration(
            "Teacher precompute + training stage",
            teacher_seconds + training_stage_seconds,
        )
        print("====================================")

    except KeyboardInterrupt:
        # 中断处理：保存当前状态为 .interrupt 并退出
        print("Training interrupted. Saving packet.")
        state_dict = packet_wrapper.state_dict()
        state_dict["train_config"] = train_config # type: ignore
        torch.save(state_dict, f"{train_config['save_path']}/{train_config['file_name']}.interrupt")
        print(f"Saved trained filler to {train_config['save_path']}/{train_config['file_name']}.interrupt")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_files_or_paths",
        type=str,
        nargs="+",
        help="Path to the training configuration file (JSON format)."
    )
    args = parser.parse_args()

    config_files_or_paths: list[str] = args.config_files_or_paths
    assert isinstance(config_files_or_paths, list)

    all_config_files: list[str] = []
    for file_or_path in config_files_or_paths:
        config_files = gather_config_files(file_or_path, pattern=r".*\.json$")
        all_config_files.extend(config_files)

    train_configs: list[TrainConfig] = [
        load_train_config(load_config_file(
            config_file, default_config_file="_default.json"
        )) for config_file in all_config_files
    ]

    print(f"Loaded {len(train_configs)} training configurations.")
    for config_file in all_config_files:
        print(f" - {config_file}")

    train_cache = TrainCache()

    for train_config in train_configs:
        train_one_config(train_config, train_cache)
