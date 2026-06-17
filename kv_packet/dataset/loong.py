from typing import Any, Iterator
import glob
import json
import os
import random
import re
import warnings
from urllib.parse import urlparse

from .abc import RetEvalEntry

# 先用短上下文英文 financial/paper 子集，以最小 generation batch 和 forward micro-batch 跑通长上下文训练流程。
# 跑通后再考虑扩大到 context_set_filter: [1, 2]、增加 num_samples，或者把 gen_batch_size / forward_batch_size 慢慢调大。

# 所有样本中只有 695 条英文样本（legal type 全部为中文样本）
# 只取 financial/paper，只取英文，只取 set=1 最短上下文档位的英文文本只有 83 条

# 梳理 loong 数据集的情况，为了暂时跑起来做了哪些取舍：
# 1) 

def loong_ret_eval_generator(
    num_samples: int,
    num_data_strs: int,
    num_shots: int,
    subset: str = "all",
    split: str = "all",
    seed: int = 42,
    **kwargs
) -> Iterator[RetEvalEntry]:
    """
    Generates evaluation entries for the Loong dataset from the official JSONL file.

    Args:
        num_samples (int): Number of samples to generate.
        num_data_strs (int): Number of data strings to include in each entry (not used here).
        num_shots (int): Number of few-shot examples to include in the preamble.
        subset (str): Domain filter. One of "all", "financial", "legal", "paper".
        split (str): Alias for domain filtering, kept for compatibility with the config interface.
        seed (int): Random seed for shuffling the dataset.
        **kwargs: Additional keyword arguments.
        - data_path (str): Path to the official loong.jsonl file.
        - doc_root (str | None): Root directory for files referenced by doc/shuffle_doc.
          Defaults to a sibling `loong_doc` directory when present, otherwise the directory containing data_path.
        - read_doc_files (bool): Whether to read doc/shuffle_doc entries as file paths. Default is True.
        - add_inst (bool): Whether to include the instruction in the task prompt. Default is True.
        - add_cot (bool): Whether to add chain-of-thought prompting to the task prompt. Default is False.
        - use_shuffle_doc (bool): Whether to use the shuffled document order if present. Default is False.
        - set_filter (str | list[str] | None): Domain values to keep: "financial", "legal", or "paper".
        - type_filter (str | list[str] | None): Alias for set_filter, kept for compatibility.
        - language_filter (str | list[str] | None): Language values to keep.
        - context_set_filter (int | str | list[int | str] | None): Loong length set values to keep.
          Set1 is roughly 10K-50K, Set2 50K-100K, Set3 100K-200K, and Set4 200K+.
        - level_filter (str | list[str] | None): Level values to keep.
        - length_filter (str | list[str] | None): Length bucket values to keep.
        - min_length_filter (int | str | None): Minimum Loong length value to keep.
        - max_length_filter (int | str | None): Maximum Loong length value to keep.
        - max_doc_chars (int | str | None): Maximum characters to keep from each loaded document.
        - max_context_chars (int | str | None): Maximum total characters to keep across all loaded documents.
          This is applied after reading the referenced files and is useful for training, where full Loong
          contexts can exceed GPU memory even when the JSONL length metadata is filtered.
        - id_filter (str | list[str] | None): Sample ids to keep.
        - exclude_id_filter (str | list[str] | None): Sample ids to exclude.

    Yields:
        RetEvalEntry: An evaluation entry containing preamble, documents, task prompt, query, and answer.

    Note:
        - Loong is released as a benchmark JSONL rather than a train/validation/test HuggingFace split.
        - The dataset is much longer than HotpotQA, MuSiQue, and 2WikiMultiHopQA, so it is usually better suited
          for evaluation than training.
    """
    data_path = kwargs.pop("data_path", "./data/loong.jsonl")
    doc_root = kwargs.pop("doc_root", None)
    read_doc_files = kwargs.pop("read_doc_files", True)
    add_inst = kwargs.pop("add_inst", True)
    add_cot = kwargs.pop("add_cot", False)
    use_shuffle_doc = kwargs.pop("use_shuffle_doc", False)
    set_filter = kwargs.pop("set_filter", None)
    type_filter = kwargs.pop("type_filter", None)
    language_filter = kwargs.pop("language_filter", None)
    context_set_filter = kwargs.pop("context_set_filter", None)
    level_filter = kwargs.pop("level_filter", None)
    length_filter = kwargs.pop("length_filter", None)
    min_length_filter = kwargs.pop("min_length_filter", None)
    max_length_filter = kwargs.pop("max_length_filter", None)
    max_doc_chars = _to_optional_int(kwargs.pop("max_doc_chars", None), "max_doc_chars")
    max_context_chars = _to_optional_int(kwargs.pop("max_context_chars", None), "max_context_chars")
    id_filter = kwargs.pop("id_filter", None)
    exclude_id_filter = kwargs.pop("exclude_id_filter", None)

    if kwargs:
        warnings.warn(f"Unused kwargs in loong_ret_eval_generator: {kwargs}")
    if num_data_strs != 0:
        warnings.warn("num_data_strs is not used for Loong; using all documents in each sample.")

    # Loong is not organized as train/validation/test. Reuse the existing
    # config fields as domain filters so configs can still follow the common
    # dataset interface: split/subset can be "all", "financial", "legal", or "paper".
    if set_filter is None:
        set_filter = _normalize_default_filter(split)
    if set_filter is None:
        set_filter = _normalize_default_filter(subset)

    entries = _load_loong_jsonl(data_path)
    doc_root = doc_root or _default_doc_root(data_path)
    doc_cache: dict[str, str] = {}
    legal_doc_cache: dict[str, dict[str, Any]] = {}
    entries = _filter_entries(
        entries,
        set_filter=set_filter,
        type_filter=type_filter,
        language_filter=language_filter,
        context_set_filter=context_set_filter,
        level_filter=level_filter,
        length_filter=length_filter,
        min_length_filter=min_length_filter,
        max_length_filter=max_length_filter,
        id_filter=id_filter,
        exclude_id_filter=exclude_id_filter,
    )

    all_indices = list(range(len(entries)))
    random.seed(seed)
    random.shuffle(all_indices)

    if num_shots > 0:
        warnings.warn("few_shot_str is not recommended for Loong")
        few_shot_indices = all_indices[:num_shots]
        all_indices = all_indices[num_shots:]
        few_shot_strs = []
        for idx in few_shot_indices:
            item = entries[idx]
            context_strs = format_data_str(
                item,
                use_shuffle_doc=use_shuffle_doc,
                doc_root=doc_root,
                read_doc_files=read_doc_files,
                doc_cache=doc_cache,
                legal_doc_cache=legal_doc_cache,
            )
            few_shot_str = f"Context: {' '.join(context_strs)}\nQuestion: {item['question']}\nAnswer: {format_answer(item['answer'])}\n"
            few_shot_strs.append(few_shot_str)
        few_shot_str = "\n".join(few_shot_strs)
    else:
        few_shot_str = ""

    if num_samples > len(all_indices):
        warnings.warn(f"num_samples ({num_samples}) is greater than dataset size ({len(all_indices)}). Reducing num_samples to dataset size.")
        num_samples = len(all_indices)

    for _ in range(num_samples):
        idx = all_indices.pop(0)
        item = entries[idx]
        question_str = item["question"]
        if add_inst and item.get("instruction"):
            question_str = f"{item['instruction']}\n{question_str}"
        if add_cot:
            question_str += "\nYou should get the final answer by thinking step by step."
        if add_inst:
            question_str += "\nYour response should end with: 'Answer: <your final answer>'.\n"

        yield RetEvalEntry(
            preamble=few_shot_str,
            documents=format_data_str(
                item,
                use_shuffle_doc=use_shuffle_doc,
                doc_root=doc_root,
                read_doc_files=read_doc_files,
                max_doc_chars=max_doc_chars,
                max_context_chars=max_context_chars,
                doc_cache=doc_cache,
                legal_doc_cache=legal_doc_cache,
            ),
            task_prompt=question_str,
            query=item["question"],
            answer=format_answer(item["answer"]),
        )


def _load_loong_jsonl(data_path: str) -> list[dict[str, Any]]:
    # The official release is a JSONL file on GitHub, not a HuggingFace dataset
    # with named splits. Keep this loader local-file based to avoid implicit
    # network access during training/evaluation.
    entries = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _filter_entries(
    entries: list[dict[str, Any]],
    set_filter: str|list[str]|None = None,
    type_filter: str|list[str]|None = None,
    language_filter: str|list[str]|None = None,
    context_set_filter: str|int|list[str|int]|None = None,
    level_filter: str|list[str]|None = None,
    length_filter: str|list[str]|None = None,
    min_length_filter: str|int|None = None,
    max_length_filter: str|int|None = None,
    id_filter: str|list[str]|None = None,
    exclude_id_filter: str|list[str]|None = None,
) -> list[dict[str, Any]]:
    # All filters accept either a single string or a list of strings. Values are
    # normalized to strings because Loong metadata such as level may be numeric
    # in some copies of the dataset.
    set_filter = _to_filter_set(set_filter)
    type_filter = _to_filter_set(type_filter)
    language_filter = _to_filter_set(language_filter)
    context_set_filter = _to_filter_set(context_set_filter)
    level_filter = _to_filter_set(level_filter)
    length_filter = _to_filter_set(length_filter)
    min_length = _to_optional_int(min_length_filter, "min_length_filter")
    max_length = _to_optional_int(max_length_filter, "max_length_filter")
    id_filter = _to_filter_set(id_filter)
    exclude_id_filter = _to_filter_set(exclude_id_filter)

    filtered = []
    for item in entries:
        if set_filter is not None and str(item.get("type")) not in set_filter:
            continue
        if type_filter is not None and str(item.get("type")) not in type_filter:
            continue
        if language_filter is not None and str(item.get("language")) not in language_filter:
            continue
        if context_set_filter is not None and str(item.get("set")) not in context_set_filter:
            continue
        if level_filter is not None and str(item.get("level")) not in level_filter:
            continue
        if length_filter is not None and str(item.get("length")) not in length_filter:
            continue
        item_length = _to_optional_int(item.get("length"), "length")
        if min_length is not None and (item_length is None or item_length < min_length):
            continue
        if max_length is not None and (item_length is None or item_length > max_length):
            continue
        if id_filter is not None and str(item.get("id")) not in id_filter:
            continue
        if exclude_id_filter is not None and str(item.get("id")) in exclude_id_filter:
            continue
        filtered.append(item)
    return filtered


def _normalize_default_filter(value: str) -> str|None:
    if value in ("all", "default", ""):
        return None
    return value


def _to_filter_set(value: str|int|list[str|int]|None) -> set[str]|None:
    if value is None:
        return None
    if isinstance(value, str|int):
        return {str(value)}
    return {str(item) for item in value}

def _to_optional_int(value: Any, name: str) -> int|None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer-compatible value, got {value!r}.") from exc

def format_data_str(
    item: dict[str, Any],
    use_shuffle_doc: bool = False,
    doc_root: str = ".",
    read_doc_files: bool = True,
    max_doc_chars: int|None = None,
    max_context_chars: int|None = None,
    doc_cache: dict[str, str]|None = None,
    legal_doc_cache: dict[str, dict[str, Any]]|None = None,
) -> list[str]:
    # The official samples include both the original document order and an optional shuffled order. 
    # Default to the original order; shuffled documents can be enabled to test order sensitivity.
    shuffle_doc = item.get("shuffle_doc")
    doc_refs = shuffle_doc if use_shuffle_doc and isinstance(shuffle_doc, list) else item.get("doc")
    if doc_refs is None:
        return []
    if isinstance(doc_refs, str):
        doc_refs = [doc_refs]
    if not read_doc_files:
        return [str(doc_ref) for doc_ref in doc_refs]

    # In the official JSONL, doc/shuffle_doc entries are references rather than
    # the document text itself: paper uses .md filenames, financial uses company
    # names or .txt filenames, and legal uses keys in legal/legal.json.
    documents = [
        _read_document(
            str(doc_ref),
            doc_type=str(item.get("type")),
            doc_root=doc_root,
            doc_cache=doc_cache,
            legal_doc_cache=legal_doc_cache,
        )
        for doc_ref in doc_refs
    ]

    return _limit_documents(
        documents,
        max_doc_chars=max_doc_chars,
        max_context_chars=max_context_chars,
    )

def _limit_documents(
    documents: list[str],
    max_doc_chars: int|None = None,
    max_context_chars: int|None = None,
) -> list[str]:
    # Loong contexts are intentionally long. Training KV Packet requires a
    # full forward/backward pass over the wrapped documents, so configs may
    # need to cap the loaded text even when evaluation uses the full context.
    if max_doc_chars is not None:
        documents = [document[:max_doc_chars] for document in documents]
    if max_context_chars is None:
        return documents

    limited_documents = []
    remaining_chars = max_context_chars
    for document in documents:
        if remaining_chars <= 0:
            break
        limited_document = document[:remaining_chars]
        if limited_document:
            limited_documents.append(limited_document)
        remaining_chars -= len(limited_document)
    return limited_documents


def _read_document(
    doc_ref: str,
    doc_type: str,
    doc_root: str,
    doc_cache: dict[str, str]|None = None,
    legal_doc_cache: dict[str, dict[str, Any]]|None = None,
) -> str:
    if doc_type == "legal":
        return _read_legal_document(doc_ref, doc_root, legal_doc_cache)
    if doc_type in ("financial", "paper"):
        return _read_document_file(doc_ref, doc_root, doc_type, doc_cache)
    raise ValueError(f"Unknown Loong document type: {doc_type}")


def _read_document_file(
    doc_ref: str,
    doc_root: str,
    doc_type: str,
    doc_cache: dict[str, str]|None = None,
) -> str:
    path = _resolve_doc_path(doc_ref, doc_root, doc_type)
    if doc_cache is not None and path in doc_cache:
        return doc_cache[path]

    with open(path, encoding="utf-8") as f:
        content = f.read()
    if doc_cache is not None:
        doc_cache[path] = content
    return content


def _read_legal_document(
    doc_ref: str,
    doc_root: str,
    legal_doc_cache: dict[str, dict[str, Any]]|None = None,
) -> str:
    legal_docs = _load_legal_docs(doc_root, legal_doc_cache)
    if doc_ref not in legal_docs:
        raise KeyError(f"Loong legal document {doc_ref!r} not found in legal.json.")

    item = legal_docs[doc_ref]
    content = item.get("content", "")
    metadata = [
        str(item.get(key))
        for key in ("case", "sub_case", "court", "legal_type", "number")
        if item.get(key)
    ]
    title = f"标题：{doc_ref}"
    meta = f"元信息：{'；'.join(metadata)}" if metadata else ""
    return "\n".join(part for part in (title, meta, content) if part)


def _load_legal_docs(
    doc_root: str,
    legal_doc_cache: dict[str, dict[str, Any]]|None = None,
) -> dict[str, Any]:
    cache_key = "legal.json"
    if legal_doc_cache is not None and cache_key in legal_doc_cache:
        return legal_doc_cache[cache_key]

    candidates = [
        os.path.join(doc_root, "legal", "legal.json"),
        os.path.join(doc_root, "legal.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                docs = json.load(f)
            if legal_doc_cache is not None:
                legal_doc_cache[cache_key] = docs
            return docs

    raise FileNotFoundError(f"Loong legal.json not found. Tried: {candidates}")


def _resolve_doc_path(doc_ref: str, doc_root: str, doc_type: str) -> str:
    # Loong documents use different storage conventions by domain:
    # paper refs are markdown filenames
    # financial refs are company names or txt filenames
    # legal refs are handled through legal.json above.
    parsed_path = urlparse(doc_ref).path
    basename = os.path.basename(parsed_path or doc_ref)
    normalized_ref = parsed_path.lstrip("/") if parsed_path else doc_ref
    extension = ".md" if doc_type == "paper" else ".txt"
    domain_dir = doc_type if doc_type in {"financial", "paper"} else ""

    candidates = []
    if os.path.isabs(doc_ref):
        candidates.append(doc_ref)
    else:
        candidates.append(doc_ref)
        candidates.append(f"{doc_ref}{extension}")
        candidates.append(os.path.join(doc_root, doc_ref))
        candidates.append(os.path.join(doc_root, f"{doc_ref}{extension}"))
        candidates.append(os.path.join(doc_root, normalized_ref))
        candidates.append(os.path.join(doc_root, f"{normalized_ref}{extension}"))
        if domain_dir:
            candidates.append(os.path.join(doc_root, domain_dir, doc_ref))
            candidates.append(os.path.join(doc_root, domain_dir, f"{doc_ref}{extension}"))
            candidates.append(os.path.join(doc_root, domain_dir, normalized_ref))
            candidates.append(os.path.join(doc_root, domain_dir, f"{normalized_ref}{extension}"))
            candidates.append(os.path.join(doc_root, domain_dir, basename))
        candidates.append(os.path.join(doc_root, basename))

    for path in candidates:
        if os.path.isfile(path):
            return path

    if doc_type == "financial":
        # Chinese financial reports often add stock-code/year prefixes, e.g.
        # `report_000651-2023-格力电器-2023年一季度报告.txt`, while the
        # JSONL doc entry is only `格力电器-2023年一季度报告`.
        matched_paths = glob.glob(os.path.join(doc_root, "financial", f"*{doc_ref}*.txt"))
        if matched_paths:
            matched_paths.sort()
            return matched_paths[-1]

    raise FileNotFoundError(
        f"Loong document file not found for doc entry {doc_ref!r}. "
        f"Tried: {candidates}. Set data_kwargs.doc_root to the loong_doc directory."
    )


def _default_doc_root(data_path: str) -> str:
    data_dir = os.path.dirname(data_path) or "."
    loong_doc_dir = os.path.join(data_dir, "loong_doc")
    if os.path.isdir(loong_doc_dir):
        return loong_doc_dir
    return data_dir


def format_answer(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    return json.dumps(answer, ensure_ascii=False)


def loong_answer_postprocess(pred_answer: str, gold_answer: str) -> tuple[str, str]:
    parts = re.split(r'(?i)Answer\s*:', pred_answer)
    if len(parts) > 1:
        pred_answer = parts[-1].lower().strip().rstrip('.')
    else:
        pred_answer = pred_answer.lower().strip().rstrip('.')

    gold_answer = gold_answer.lower().strip().rstrip('.')
    return pred_answer, gold_answer