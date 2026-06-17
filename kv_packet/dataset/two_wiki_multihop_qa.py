from typing import Iterator
import random
import re
import warnings

from datasets import Dataset, load_dataset

from .abc import RetEvalEntry


def two_wiki_multihop_qa_ret_eval_generator(
    num_samples: int,
    num_data_strs: int,
    num_shots: int,
    subset: str = "default",
    split: str = "validation",
    seed: int = 42,
    **kwargs
) -> Iterator[RetEvalEntry]:
    """
    Generates evaluation entries for the 2WikiMultiHopQA dataset. Each document
    is the concatenation of sentences from one context paragraph.

    Args:
        num_samples (int): Number of samples to generate.
        num_data_strs (int): Number of data strings to include in each entry (not used here).
        num_shots (int): Number of few-shot examples to include in the preamble.
        subset (str): Subset of the 2WikiMultiHopQA dataset to use.
        split (str): Split of the dataset to use ("train", "validation", or "test").
        seed (int): Random seed for shuffling the dataset.
        **kwargs: Additional keyword arguments.
        - add_inst (bool): Whether to add instructions to the question prompt. Default is True.
        - add_cot (bool): Whether to add chain-of-thought prompting to the question prompt. Default is True.
        - question_type (list[str] | None): Question types to keep. Default is None.

    Yields:
        RetEvalEntry: An evaluation entry containing preamble, documents, task prompt, query, and answer.

    Note:
        - The 2WikiMultiHopQA dataset should be used by an instruction-tuned model as it requires multi-hop reasoning.
        - Few-shot prompting is not recommended for this dataset as the multi-hop content is missing in the context.
        - num_data_strs is not used in this generator since each sample already contains its own context paragraphs.
    """
    add_inst = kwargs.pop("add_inst", True)
    add_cot = kwargs.pop("add_cot", True)
    # 支持可选 question_type 过滤问题类型，2WikiMultiHopQA 包含 "comparison" 和 "bridge" 两种问题类型
    question_type: list[str]|str|None = kwargs.pop("question_type", None)
    if isinstance(question_type, str):
        question_type = [question_type]

    if kwargs:
        warnings.warn(f"Unused kwargs in two_wiki_multi_hop_qa_ret_eval_generator: {kwargs}")
    if num_data_strs != 0:
        warnings.warn("num_data_strs is not used for 2WikiMultiHopQA; using all context paragraphs in each sample.")

    ds_split = load_dataset("framolfese/2WikiMultihopQA", subset, split=split)
    assert isinstance(ds_split, Dataset)

    all_indices = list(range(len(ds_split)))
    random.seed(seed)
    random.shuffle(all_indices)
    if question_type is not None:
        all_indices = [idx for idx in all_indices if ds_split[idx]["type"] in question_type]

    def format_data_str(index: int) -> list[str]:
        item = ds_split[index]
        context = item["context"]
        sentences = context["sentences"]
        return ["".join(sentence_group) for sentence_group in sentences]

    if num_shots > 0:
        warnings.warn("few_shot_str is not recommended for 2WikiMultiHopQA")
        few_shot_indices, all_indices = next_valid_samples(
            ds_split,
            all_indices,
            num_shots,
            question_type,
        )
        few_shot_strs = []
        for idx in few_shot_indices:
            item = ds_split[idx]
            question = item["question"]
            answer = item["answer"]
            context_strs = format_data_str(idx)
            few_shot_str = f"Context: {' '.join(context_strs)}\nQuestion: {question}\nAnswer: {answer}\n"
            few_shot_strs.append(few_shot_str)
        few_shot_str = "\n".join(few_shot_strs)
    else:
        few_shot_str = ""

    if num_samples > len(all_indices):
        warnings.warn(f"num_samples ({num_samples}) is greater than dataset size ({len(all_indices)}). Reducing num_samples to dataset size.")
        num_samples = len(all_indices)

    for _ in range(num_samples):
        idx, all_indices = next_valid_sample(ds_split, all_indices, question_type)
        item = ds_split[idx]
        question_str = item["question"]
        if add_inst:
            question_str = f"Answer the following question based on the provided context.\n{question_str}"
        if add_cot:
            question_str += "\nYou should get the final answer by thinking step by step.\n"
        if add_inst:
            question_str += "\nYour response should end with: 'Short Answer: <your final answer>'.\n"
        # if add_cot:
        #     question_str += "You should get the final answer by thinking step by step.\n"
        # if add_inst:
        #     question_str += "Your response should end with: 'Short Answer: <your final answer>'.\n"

        yield RetEvalEntry(
            preamble=few_shot_str,
            documents=format_data_str(idx),
            task_prompt=question_str,
            query=item["question"],
            answer=item["answer"],
        )


def next_valid_sample(
    ds: Dataset,
    indices: list[int],
    question_type: list[str]|None = None,
) -> tuple[int, list[int]]:
    counter = 0
    while counter < len(indices):
        idx = indices[counter]
        sample = ds[idx]
        if question_type is None or sample["type"] in question_type:
            return idx, indices[counter + 1 :]
        counter += 1
    raise ValueError("No valid sample found with the specified question types.")


def next_valid_samples(
    ds: Dataset,
    indices: list[int],
    num_samples: int,
    question_type: list[str]|None = None,
) -> tuple[list[int], list[int]]:
    valid_indices = []
    remaining_indices = indices
    for _ in range(num_samples):
        idx, remaining_indices = next_valid_sample(ds, remaining_indices, question_type)
        valid_indices.append(idx)
    return valid_indices, remaining_indices


def two_wiki_multihop_qa_answer_postprocess(pred_answer: str, gold_answer: str) -> tuple[str, str]:
    parts = re.split(r'(?i)Answer\s*:', pred_answer)
    # 只要 raw output 里没有出现 Answer: 或 Short Answer:，最终 pred 就会变成 ""
    if len(parts) > 1:
        pred_answer = parts[-1].lower().strip().rstrip('.')
    else:
        pred_answer = ""

    gold_answer = gold_answer.lower().strip().rstrip('.')
    return pred_answer, gold_answer
