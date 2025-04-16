import torch
import spacy
from transformers import pipeline

classifier = pipeline("zero-shot-classification")


def extract_user_preferences(message: str):
    doc = nlp(message)
    results = []

    noise_words = {"lot", "something", "things", "thing", "time", "do"}
    negative_phrases = ["don't like", "do not like", "hate", "dislike"]
    positive_phrases = ["like", "love", "prefer"]
    candidate_labels = ["food", "workout", "hobby", "entertainment", "lifestyle", "other"]

    def get_category(word):
        result = classifier(word, candidate_labels=candidate_labels, multi_label=False)
        return result["labels"][0] if result["labels"] else "other"

    for sent in doc.sents:
        sent_text = sent.text.lower()

        sentiment = None
        if any(neg in sent_text for neg in negative_phrases):
            sentiment = "dislike"
        elif any(pos in sent_text for pos in positive_phrases):
            sentiment = "like"

        if sentiment:
            for token in sent:
                if token.pos_ in ["NOUN", "PROPN", "VERB"]:
                    lemma = token.lemma_.lower()
                    if lemma not in noise_words and lemma not in positive_phrases + negative_phrases:
                        category = get_category(lemma)
                        results.append({
                            "value": lemma,
                            "sentiment": sentiment,
                            "category": category
                        })

    return results


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)


nlp = spacy.load("en_core_web_sm")


def extract_user_facts(message: str):
    doc = nlp(message)
    facts = {}

    for ent in doc.ents:
        if ent.label_ == "QUANTITY" or ent.label_ == "CARDINAL":
            if "cm" in ent.text or "centimeter" in ent.text:
                facts["height"] = ent.text
            elif "kg" in ent.text or "kilogram" in ent.text:
                facts["weight"] = ent.text

    if "goal" in message.lower():
        for sent in doc.sents:
            if "goal" in sent.text.lower():
                facts["goal"] = sent.text.split("is")[-1].strip(". ")

    days = []
    for token in doc:
        if token.text.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            days.append(token.text)
    if days:
        facts["gym_days"] = days

    # intermediate expert level Todo
    return facts
