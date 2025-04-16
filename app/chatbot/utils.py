import torch
import spacy


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

    return facts
