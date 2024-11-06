from typing import List, Optional
import re
from datasets import load_dataset
import spacy
from tner import TransformersNER


model = TransformersNER("tner/roberta-large-wnut2017")
nlp = spacy.load("en_core_web_sm")


def find_entities(text: str, check_pronoun=False) -> List[str]:
    pred = model.predict([text])
    pred = pred['entity_prediction'][0]
    named_entities = [" ".join(p['entity']) for p in pred if p['type'] == "person"]
    if named_entities:
        return named_entities
    if not named_entities and check_pronoun:
        if ' he ' in text:
            return ["he"]
        if ' she ' in text:
            return ["she"]
    return []


def find_subject(text: str, subject_of_interest: List[str]) -> Optional[str]:
    gsm8k_question_keywords = ['how', 'what', 'calculate', 'determine', 'when', 'find']
    keyword = None
    for kw in gsm8k_question_keywords:
        if kw in text.lower():
            keyword = kw
            break

    if keyword is None:
        return None

    keyword_index = text.lower().rindex(keyword)

    text_after_keyword = text[keyword_index:]
    closest_interest = None
    closest_distance = float("inf")
    for s in subject_of_interest:
        if s not in text_after_keyword:
            continue
        idx = text_after_keyword.index(s)
        if idx < closest_distance:
            closest_interest = s
            closest_distance = idx
    return closest_interest


def create_named_template(e):
    e["question_named_template"] = None
    e["first_entity"] = None
    e["entities_in_question"] = None

    question_sentences = e['question'].split('. ')
    if len(question_sentences) < 2:
        return e
    question = question_sentences[-1].strip()
    if not question.endswith("?"):
        return e
    #
    try:
        entities_in_question = find_entities(question, check_pronoun=True)
        if not entities_in_question:
            return e
        entities_in_question = [re.sub(r"['’]s$", "", e.strip()) for e in entities_in_question]
        entities_in_question = [re.sub(r",$", "", e.strip()) for e in entities_in_question]

        subj_entity_in_question = find_subject(question, entities_in_question)
        if subj_entity_in_question is None:
            return e

        named_template_last = re.sub(rf"\b{subj_entity_in_question}\b", "{Name2}", question)
        #
        if subj_entity_in_question in ["he", "she"]:
            first_entity = find_entities(question_sentences[0].strip(), check_pronoun=False)
            if first_entity is None and len(question_sentences) >= 3:
                first_entity = find_entities(question_sentences[1].strip(), check_pronoun=False)
            if not first_entity:
                return e
            first_entity = first_entity[0]
        else:
            first_entity = subj_entity_in_question

        sentences_before_last = question_sentences[:-1]
        named_template_before_last = [re.sub(rf"\b{first_entity}\b", "{Name1}", s) for s in sentences_before_last]
        #
        question_named_template = ". ".join(named_template_before_last + [named_template_last])
        if "{Name1}" not in question_named_template or "{Name2}" not in question_named_template:
            return e

        e["question_named_template"] = ". ".join(named_template_before_last + [named_template_last])
        e["first_entity"] = first_entity
        e["entities_in_question"] = subj_entity_in_question
        return e
    except re.error:
        return e


if __name__ == "__main__":
    ds = load_dataset("openai/gsm8k", "main")
    ds = ds.map(create_named_template)
    ds = ds.filter(lambda e: e["question_named_template"] is not None)
    ds.save_to_disk("gsm8k-who")
    ds.push_to_hub("kenhktsui/gsm8k-who", private=True)
