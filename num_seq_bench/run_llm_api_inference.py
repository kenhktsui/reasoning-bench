import os.path
from typing import List, Callable, Optional, Set
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd
from datasets import load_dataset, Dataset
from utils.llm_client import ask_llm


def decode_text(text: str) -> dict:
    res_list = re.findall(r'\\?"answer\\?": ([-\d,\\"]+|null)', text)
    if res_list:
        result = res_list[0]
        if "," in result:
            result = "".join([char for char in list(result) if char != ","])
        if result == "null":
            return {"pred_answer": None, "pred_is_answerable": False}
        else:
            try:
                return {"pred_answer": int(result), "pred_is_answerable": True}
            except ValueError:
                return {"pred_answer": None, "pred_is_answerable": None}
    return {"pred_answer": None, "pred_is_answerable": None}



def prompt_llm_test_data(
        test_dataset: Dataset,
        filename: str,
        model_name_list: List[str],
        ask_llm: Callable[[str, str], dict],
        num_threads: int = 16,
        processed_line: Optional[Set[str]] = None
    ):
    file_lock = threading.Lock()

    def process_and_write(d: dict, model_name: str):
        result = ask_llm(d["question"], model_name)
        result_dict = {}
        result_dict["raw_response"] = result
        result_dict["prompt"] = d["question"]
        result_dict["id"] = d["id"]
        result_dict["question_type"] = d["question_type"]
        result_dict["sequence_type"] = d["sequence_type"]
        result_dict["visible_length"] = d["visible_length"]
        result_dict["nth_element"] = d["nth_element"]
        result_dict["model_name"] = model_name
        result_dict["ground_truth"] = d["answer"]

        if result:
            with file_lock:
                with open(filename, 'a') as f:
                    json.dump(result_dict, f)
                    f.write('\n')
                    f.flush()
        return result

    # calling LLM models concurrently
    for d in tqdm(test_dataset, desc="Text sample"):
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_and_write, d, model_name) for model_name in model_name_list
                       if processed_line is not None and model_name + '__' + str(d['id']) not in processed_line]
            for _ in tqdm(as_completed(futures), total=len(model_name_list)):
                pass


if __name__ == "__main__":
    import json

    MODEL_NAME_LIST = [
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "microsoft/WizardLM-2-8x22B",
        "microsoft/WizardLM-2-7B",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mistral-large-latest",
        "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
        "databricks/dbrx-instruct",
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "claude-3-5-sonnet-20240620",
        "claude-3-haiku-20240307",
        "gpt-4o",
        "deepseek-chat",
        "deepseek-coder",
        "yi-large",
        "yi-large-turbo",
        "reka-core-20240501",
        "reka-flash-20240226",
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ]

    result_path = "data/inductive_reasoning_numeric_test_data_result_cleaned.json"
    result_analysis_path = "data/inductive_reasoning_numeric_test_data_result_cleaned_analysis.json"

    def rerun():
        is_rerun = input(f"{result_path} is found. Do you want to run the unprocessed line? (y/n)")
        if is_rerun == "y":
            return True
        if is_rerun == "n":
            return False

    is_rerun = False
    if (os.path.exists(result_path) and rerun()) or not os.path.exists(result_path):
        is_rerun = True

    if is_rerun:
        dataset = load_dataset("kenhktsui/num_seq_bench", split='train')
        print(dataset)

        processed_line = set()
        if os.path.exists(result_path):
            with open(result_path) as f:
                for l in f:
                    l = json.loads(l)
                    processed_line.add(l['model_name'] + '__' + str(l['id']))
            print("No of processed line: ", len(processed_line))

        prompt_llm_test_data(
            dataset,
            result_path,
            MODEL_NAME_LIST,
            ask_llm,
            num_threads=16,
            processed_line=processed_line
        )
        with open(result_path) as f:
            result_dict_list = [json.loads(l) for l in f]

        print(f"{len(result_dict_list)} lines processed vs target {len(MODEL_NAME_LIST) * len(dataset)}")
