from datasets import load_dataset


def create_no_q(e):
    question_sentences = e['question'].split('. ')
    e["no_question"] = '. '.join(question_sentences[:-1]) + '.'
    return e

def no_answer(e):
    e["dont_answer"] = e["no_question"] + " Don't answer."
    return e


if __name__ == "__main__":
    ds = load_dataset("openai/gsm8k", "main")
    ds = ds.map(create_no_q)
    ds = ds.map(no_answer)
    ds.push_to_hub("kenhktsui/gsm8k-noq", private=True)
