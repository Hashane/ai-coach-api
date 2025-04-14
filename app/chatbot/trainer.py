import json
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import os


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element = token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def train_embeddings(data_path='data/data.json',
                     embedding_output='sbert_embeddings.pkl',
                     label_output='label_encoder.pkl',
                     responses_output='responses_dict.pkl',
                     model_name='jgammack/distilbert-base-mean-pooling',
                     model_dir='/Users/judehashane/PycharmProjects/AICoachApiProject/app/chatbot'):
    os.chdir(model_dir)

    # Load data
    with open(data_path, "r") as file:
        data = json.load(file)

    patterns = []
    tags = []
    responses_dict = {}

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            patterns.append(pattern.lower())
            tags.append(intent["tag"])
        responses_dict[intent["tag"]] = intent["responses"]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and get embeddings
    inputs = tokenizer(patterns, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)

    embeddings = mean_pooling(model_output, inputs['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    X = embeddings.cpu().numpy()

    # Encode tags
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(tags)

    # Mean per tag
    unique_tags = list(label_encoder.classes_)
    tag_to_embeddings = {tag: [] for tag in unique_tags}
    for i, tag in enumerate(tags):
        tag_to_embeddings[tag].append(X[i])

    final_embeddings = []
    final_labels = []
    for tag, embs in tag_to_embeddings.items():
        mean_emb = np.mean(embs, axis=0)
        final_embeddings.append(mean_emb)
        final_labels.append(tag)

    # Save results
    with open(embedding_output, "wb") as f:
        pickle.dump(np.array(final_embeddings), f)
    with open(label_output, "wb") as f:
        pickle.dump(label_encoder, f)
    with open(responses_output, "wb") as f:
        pickle.dump(responses_dict, f)

    print("✅ Embeddings trained and saved successfully.")

# Example usage:
# train_embeddings()
