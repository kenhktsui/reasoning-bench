from typing import List
import re
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


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
SEQUENCE_TYPE_LIST = ["arithmetic", "geometric", "fibonacci", "quadratic", "triangular",
                      "prime", "power", "factorial", "alternating",
                      "monotonic_random"
                      ]
QUESTION_TYPE_LIST = ["next", "nth", "previous"]
VISLEN_LIST = [5, 6, 7, 8]
INFERENCELEN_LIST = range(2, 16)


def decode_text(text: str) -> dict:
    if text.strip() == "null":
        return {"pred_answer": "null"}

    res_list = re.findall(r'\\?"answer\\?":\s*("null"|null|[-\d,\\"]+|"[-\d,\\"]+")', text)
    if res_list:
        result = res_list[0]
        result = result.strip().strip('"').strip("'")
        if "," in result:
            result = "".join([char for char in list(result) if char != ","])
        if result == "null":
            return {"pred_answer": "null"}
        else:
            try:
                return {"pred_answer": int(result)}
            except ValueError:
                return {"pred_answer": None}
    return {"pred_answer": None}


def change_monotonic_random_ans(s: dict) -> None:
    if s['sequence_type'] == 'monotonic_random':
        s['ground_truth'] = 'null'


def calculate_inference_length(s: dict) -> None:
    s['inference_length'] = s['nth_element'] - s['visible_length'] + 1


def get_metrics(result_dict_list: List[dict], mode="string_match_fallback") -> dict:
    it_following_error = 0
    n_total = 0
    n_correct_ans = 0
    not_answerable_actual = []
    not_answerable_pred = []
    for i in result_dict_list:
        is_random_seq = i["sequence_type"] == "monotonic_random"
        if mode == "ignore_format_following_error":
            if i["pred_answer"] is None:
                it_following_error += 1
            else:
                n_total += 1
                not_answerable_actual.append(is_random_seq)
                not_answerable_pred.append(i["pred_answer"] == "null")
                if (is_random_seq and i["pred_answer"] == "null") or i["ground_truth"] == i["pred_answer"]:
                    n_correct_ans += 1
        elif mode == "capture_format_following_error":
            n_total += 1
            not_answerable_actual.append(is_random_seq)
            not_answerable_pred.append(i["pred_answer"] == "null")
            if i["pred_answer"] is None:
                it_following_error += 1
            elif (is_random_seq and i["pred_answer"] == "null") or i["ground_truth"] == i["pred_answer"]:
                n_correct_ans += 1
        elif mode == "string_match_fallback":
            if i["pred_answer"] is None:
                it_following_error += 1

            last_number = None
            all_number = re.findall(r'(\d+)', i['raw_response'])
            if all_number:
                last_number = all_number[-1].strip('"').strip("'").strip(",")
                try:
                    last_number = int(last_number)
                except ValueError:
                    pass
            n_total += 1
            not_answerable_actual.append(is_random_seq)
            not_answerable_pred.append(i["pred_answer"] == "null" or "null" in i['raw_response'])
            if is_random_seq and (i["pred_answer"] == "null" or "null" in i['raw_response']):
                n_correct_ans += 1
            elif i["ground_truth"] == i["pred_answer"] or (last_number and i["ground_truth"] == last_number):
                n_correct_ans += 1
        else:
            raise NotImplementedError("ignore_format_following_error|capture_format_following_error|string_match are available")
    return {
        "accuracy": n_correct_ans/n_total,
        "n_total": n_total,
        "it_following_error": it_following_error,
        "abstain_f1": f1_score(not_answerable_actual, not_answerable_pred, zero_division=0.0),
        "abstain_precision": precision_score(not_answerable_actual, not_answerable_pred, zero_division=0.0),
        "abstain_recall": recall_score(not_answerable_actual, not_answerable_pred, zero_division=0.0)
    }


def construct_pandas_df(result_dict_list, column_name, value="accuracy"):
    column_map = {
        "question_type": QUESTION_TYPE_LIST,
        "sequence_type": SEQUENCE_TYPE_LIST,
        "visible_length": VISLEN_LIST,
        "inference_length": INFERENCELEN_LIST
    }
    df_dict = {}
    for t in column_map[column_name]:
        if t not in df_dict:
            df_dict[t] = {}
        for m in MODEL_NAME_LIST:
            filter_cond = lambda x: True
            if column_name == "inference_length":
                filter_cond = lambda x: x['question_type'] == 'nth'
            df_dict[t][m] = get_metrics([i for i in result_dict_list
                                         if i['model_name'] == m and i[column_name] == t and filter_cond(i)
                                         ])[value]

    return pd.DataFrame.from_dict(df_dict)


if __name__ == "__main__":
    import json

    RESULT_PATH = "data/inductive_reasoning_numeric_test_data_result_cleaned.json"

    # loading LLM response and doing preprocessing
    with open(RESULT_PATH) as f:
        result_dict_list = [json.loads(l) for l in f]

    for s in result_dict_list:
        parsed = decode_text(s['raw_response'])
        s.update(parsed)
        change_monotonic_random_ans(s)
        calculate_inference_length(s)


    # Accuracy By Question Type
    task_list_df = construct_pandas_df(result_dict_list, "question_type", "accuracy")
    task_list_df['macro avg'] = task_list_df[['next', 'nth', 'previous']].mean(1)
    print("Accuracy By Question Type")
    print(task_list_df.sort_values('macro avg', ascending=False).to_markdown())

    # Abstain F1
    abstain_f1_dict = {"f1": {}, "precision": {}, "recall": {}}
    for m in MODEL_NAME_LIST:
        random_seq = [i for i in result_dict_list if i['model_name'] == m and i['sequence_type'] == 'monotonic_random']
        abstain_f1_dict["precision"][m] = get_metrics(random_seq)["abstain_precision"]
        abstain_f1_dict["recall"][m] = get_metrics(random_seq)["abstain_recall"]
        abstain_f1_dict["f1"][m] = get_metrics(random_seq)["abstain_f1"]
    abstain_f1_df = pd.DataFrame.from_dict(abstain_f1_dict)
    print("Abstain F1")
    print(abstain_f1_df.sort_values('f1', ascending=False).to_markdown())

    # Accuracy By Sequence Type
    seq_list_df = construct_pandas_df(result_dict_list, "sequence_type", "accuracy")
    seq_list_df['macro avg'] = seq_list_df.mean(1)
    print("Accuracy By Sequence Type")
    print(seq_list_df.sort_values('macro avg', ascending=False).to_markdown())

    # Accuracy By Visible Length
    vislen_list_df = construct_pandas_df(result_dict_list, "visible_length", "accuracy")
    print("Accuracy By Visible Length")
    print(vislen_list_df.to_markdown())

    # Accuracy By Inference Length
    nth_list_df = construct_pandas_df(result_dict_list, "inference_length", "accuracy")
    print("Accuracy By Inference Length")
    print(nth_list_df.to_markdown())
