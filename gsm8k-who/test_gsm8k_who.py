import random
import os.path
from typing import List, Callable, Optional, Set
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset, Dataset
import sys
sys.path.append("..")
from utils.llm_client import ask_llm


random.seed(42)


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
        result_dict = {}

        standard_result = ask_llm(d["standard_question"], model_name)
        random_result = ask_llm(d["random_question"], model_name)
        sibling_result = ask_llm(d["sibling_question"], model_name)

        result_dict["standard_question"] = d["standard_question"]
        result_dict["random_question"] = d["random_question"]
        result_dict["sibling_question"] = d["sibling_question"]
        result_dict["standard_result"] = standard_result
        result_dict["random_result"] = random_result
        result_dict["sibling_result"] = sibling_result
        result_dict["model_name"] = model_name

        with file_lock:
            with open(filename, 'a') as f:
                json.dump(result_dict, f)
                f.write('\n')
                f.flush()

    # calling LLM models concurrently
    for d in tqdm(test_dataset, desc="Text sample"):
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_and_write, d, model_name) for model_name in model_name_list
                       if processed_line is not None and model_name + '__' + d['standard_question'] not in processed_line]
            for _ in tqdm(as_completed(futures), total=len(model_name_list)):
                pass


if __name__ == "__main__":
    import json

    MODEL_NAME_LIST = [
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        # "deepseek-chat",
        # "deepseek-coder",
        # "yi-large",
        # "yi-large-turbo",
        # "reka-core-20240501",
        # "reka-flash-20240226",
        # "gemini-1.5-pro",
        # "gemini-1.5-flash"
    ]

    result_path = "data/result.json"

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
        dataset = load_dataset("kenhktsui/gsm8k-who-test", split='train')
        print(dataset)

        processed_line = set()
        if os.path.exists(result_path):
            with open(result_path) as f:
                for l in f:
                    l = json.loads(l)
                    processed_line.add(l['model_name'] + '__' + l['standard_question'])
            print("No of processed line: ", len(processed_line))

        prompt_llm_test_data(
            dataset,
            result_path,
            MODEL_NAME_LIST,
            ask_llm,
            num_threads=40,
            processed_line=processed_line
        )
        with open(result_path) as f:
            result_dict_list = [json.loads(l) for l in f]

        print(f"{len(result_dict_list)} lines processed vs target {len(MODEL_NAME_LIST) * len(dataset)}")
