from typing import Iterator
import random
import re
import warnings

from datasets import Dataset, load_dataset

from .abc import RetEvalEntry


def musique_ret_eval_generator(
    num_samples: int,
    num_data_strs: int,
    num_shots: int,
    subset: str = "default",
    split: str = "validation",
    seed: int = 42,
    **kwargs
) -> Iterator[RetEvalEntry]:
    """
    Generates evaluation entries for the MuSiQue dataset. Each document is one
    paragraph from the sample context.

    Args:
        num_samples (int): Number of samples to generate.
        num_data_strs (int): Number of data strings to include in each entry (not used here).
        num_shots (int): Number of few-shot examples to include in the preamble.
        subset (str): Subset of the MuSiQue dataset to use.
        split (str): Split of the dataset to use ("train", "validation", or "test").
        seed (int): Random seed for shuffling the dataset.
        **kwargs: Additional keyword arguments.
        - add_inst (bool): Whether to add instructions to the question prompt. Default is True.
        - add_cot (bool): Whether to add chain-of-thought prompting to the question prompt. Default is True.
        - answer_only (bool): Whether to explicitly suppress reasoning text and require only Short Answer.
          Default is True.
        - answerable_only (bool): Whether to skip unanswerable samples. Default is True.

    Yields:
        RetEvalEntry: An evaluation entry containing preamble, documents, task prompt, query, and answer.

    Note:
        - The MuSiQue dataset should be used by an instruction-tuned model as it requires multi-hop reasoning.
        - Few-shot prompting is not recommended for this dataset as the multi-hop content is missing in the context.
        - num_data_strs is not used in this generator since each sample already contains its own paragraphs.
    """
    add_inst = kwargs.pop("add_inst", True)
    add_cot = kwargs.pop("add_cot", True)
    answer_only = kwargs.pop("answer_only", True)
    # MuSiQue 里有些样本是不可回答的，过滤可回答样本
    answerable_only = kwargs.pop("answerable_only", True)  

    if kwargs:
        warnings.warn(f"Unused kwargs in musique_ret_eval_generator: {kwargs}")
    if num_data_strs != 0:
        warnings.warn("num_data_strs is not used for MuSiQue; using all paragraphs in each sample.")

    ds_split = load_dataset("awinml/musique", subset, split=split)
    assert isinstance(ds_split, Dataset)

    all_indices = list(range(len(ds_split)))
    random.seed(seed)
    random.shuffle(all_indices)
    if answerable_only:
        all_indices = [idx for idx in all_indices if ds_split[idx]["answerable"]]

    def format_data_str(index: int) -> list[str]:
        item = ds_split[index]
        paragraphs = item["paragraphs"]
        return [paragraph["paragraph_text"] for paragraph in paragraphs]

    if num_shots > 0:
        warnings.warn("few_shot_str is not recommended for MuSiQue")
        few_shot_indices, all_indices = next_valid_samples(
            ds_split,
            all_indices,
            num_shots,
            answerable_only,
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
        idx, all_indices = next_valid_sample(ds_split, all_indices, answerable_only)
        item = ds_split[idx]
        if idx == 0:
            print('Loaded Musique dataset keys:', item.keys()) # checkpoint
        question_str = item["question"]
        # if add_inst:
        #     question_str = f"Answer the following question based on the provided context.\n{question_str}"
        # if add_cot and not answer_only:
        #     question_str += "\nYou should get the final answer by thinking step by step.\n"
        # if add_inst and answer_only:
        #     question_str += (
        #         "\nThink internally if needed, but do not output reasoning, explanations, or <think> content.\n"
        #         "Your entire response must be exactly: 'Short Answer: <your final answer>'.\n"
        #     )
        # elif add_inst:
        #     question_str += "\nYour response should end with: 'Short Answer: <your final answer>'.\n"

        if add_inst:
            question_str = (
                "Answer the following multi-hop question using only the provided context.\n"
                + question_str
            )

        if add_cot:
            question_str += (
                "\nReason through the necessary evidence concisely. "
                "Avoid repeating the question or reconsidering the same possibility.\n"
            )

        # question_str += (
        #     "\nAfter reasoning, you must provide the final answer on a new line "
        #     "using exactly this format:\n"
        #     "Short Answer: <your final answer>\n"
        # )

        question_str += (
            "\nReason through only the evidence needed to answer the question. "
            "Keep the reasoning concise and avoid repeating or reconsidering "
            "the same possibilities. After finishing the reasoning, provide "
            "one short final answer in exactly this format:\n"
            "Short Answer: <answer>\n"
            "The final answer must contain only the requested entity, name, "
            "number, date, place, or phrase."
        )

        answer_str = item["answer"]
        data_strs = format_data_str(idx)

        # print('In MusiQue generator, sample idx:', idx) # checkpoint
        # print('Question:', question_str) # checkpoint
        # print('Answer:', answer_str) # checkpoint
        # print('Context strings:', data_strs) # checkpoint
        # print('----------------------------') # checkpoint

        yield RetEvalEntry(
            preamble=few_shot_str,
            documents=data_strs,
            task_prompt=question_str,
            query=item["question"],
            answer=item["answer"],
        )


def next_valid_sample(
    ds: Dataset,
    indices: list[int],
    answerable_only: bool = True,
) -> tuple[int, list[int]]:
    counter = 0
    while counter < len(indices):
        idx = indices[counter]
        sample = ds[idx]
        if not answerable_only or sample["answerable"]:
            return idx, indices[counter + 1 :]
        counter += 1
    raise ValueError("No valid sample found with the specified answerable_only setting.")


def next_valid_samples(
    ds: Dataset,
    indices: list[int],
    num_samples: int,
    answerable_only: bool = True,
) -> tuple[list[int], list[int]]:
    valid_indices = []
    remaining_indices = indices
    for _ in range(num_samples):
        idx, remaining_indices = next_valid_sample(ds, remaining_indices, answerable_only)
        valid_indices.append(idx)
    return valid_indices, remaining_indices

def musique_answer_postprocess(pred_answer: str, gold_answer: str) -> tuple[str, str]:
    # Prefer the explicit final-answer marker. Matching plain "Answer:" is too
    # broad for MuSiQue because long reasoning often repeats "answer" language
    # before the model reaches the required final line.
    pred_answer = re.sub(r"(?is)<think>.*?</think>", "", pred_answer).strip()
    parts = re.split(r"(?i)Short\s+Answer\s*:", pred_answer)
    if len(parts) == 1:
        parts = re.split(r"(?im)^\s*Answer\s*:", pred_answer)

    if len(parts) > 1:
        # # 答案标记后为空时会 IndexError: list index out of range
        # pred_answer = parts[-1]
        # pred_answer = pred_answer.splitlines()[0].lower().strip().rstrip(".")
        # 答案标记后为空时会返回空答案，不影响后续 sample 的评测
        answer_lines = parts[-1].splitlines()
        pred_answer = (
            answer_lines[0].lower().strip().rstrip(".")
            if answer_lines
            else ""
        )
    else:
        pred_answer = ""

    gold_answer = gold_answer.lower().strip().rstrip('.')
    return pred_answer, gold_answer

# def musique_answer_postprocess(pred_answer: str, gold_answer: str) -> tuple[str, str]:
#     pred_answer = pred_answer.lower().strip()
#     gold_answer = gold_answer.lower().strip().rstrip(".")

#     pred_answer = re.sub(r"</s>|<\|eot_id\|>|<\|end_of_text\|>", " ", pred_answer)

#     for marker in ("<tool_call>", "<think>", "</think>"):
#         pred_answer = pred_answer.replace(marker, " ")
#     pred_answer = re.sub(r"\s+", " ", pred_answer).strip()

#     answer_markers = [
#         "short answer:",
#         "final answer:",
#         "the final answer is:",
#         "the final answer is",
#         "the answer is:",
#         "the answer is",
#         "so, the final answer would be:",
#         "so the final answer would be:",
#         "so the final answer is:",
#         "so the final answer is",
#         "answer:",
#         "answer is:",
#         "answer is",
#     ]

#     best_idx = -1
#     best_marker = None
#     for marker in answer_markers:
#         idx = pred_answer.rfind(marker)
#         if idx != -1 and idx > best_idx:
#             best_idx = idx
#             best_marker = marker

#     if best_marker is not None:
#         pred_answer = pred_answer[best_idx + len(best_marker):].strip()

#     # Some generations quote the requested answer format inside a longer sentence.
#     nested_parts = re.split(r"(?i)\bshort answer\s*:", pred_answer)
#     if len(nested_parts) > 1:
#         pred_answer = nested_parts[-1].strip()

#     pred_answer = pred_answer.split("\n", 1)[0].strip()
#     pred_answer = pred_answer.strip(" \"'").strip(" .,:;")

#     # Extract short answers from common explanatory MuSiQue generations.
#     extract_patterns = [
#         r"\balso played\s+(.+?)(?:\s+in\s+|\s+on\s+|\s+for\s+|[.;]|$)",
#         r"\bplayed\s+(.+?)(?:\s+in\s+|\s+on\s+|\s+for\s+|[.;]|$)",
#         r"\bstands for\s+(.+?)(?:[.;]|$)",
#         r"\bare the\s+(.+?)(?:[.;]|$)",
#         r"\bis the\s+(.+?)(?:[.;]|$)",
#         r"\bis located in\s+(.+?)(?:[.;]|$)",
#         r"\bwas established in\s+(\d{3,4})\b",
#         r"\bwas built in\s+(.+?)(?:[.;]|$)",
#         r"\boccurred .*?\b(\d+)\s+times\b",
#     ]
#     for pattern in extract_patterns:
#         match = re.search(pattern, pred_answer)
#         if match:
#             candidate = match.group(1).strip(" \"'.:;,()")
#             if candidate:
#                 pred_answer = candidate
#                 break

#     if re.fullmatch(r"\d+(?:\.\d+)?", gold_answer):
#         numbers = re.findall(r"\d+(?:\.\d+)?", pred_answer)
#         if numbers:
#             pred_answer = numbers[-1]

#     pred_answer = pred_answer.strip(" \"'").strip(" .,:;")

#     return pred_answer, gold_answer
