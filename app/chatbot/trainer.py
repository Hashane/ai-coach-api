import json
import pathlib
import nltk
import numpy as np
import pickle
from newspaper import Article
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

nltk.download('punkt')


class ChatbotTrainer:
    def __init__(self):
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.label_encoder = LabelEncoder()
        self.responses_dict = {}
        self.X = None
        self.y = None

    def initialize_from_json(self, data_path: str):
        """Initialize from JSON data file"""
        with open(data_path, "r") as file:
            data = json.load(file)

        patterns = []
        tags = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                patterns.append(pattern.lower())
                tags.append(intent["tag"])
            self.responses_dict[intent["tag"]] = intent["responses"]

        # Convert text to embeddings
        X = self.sbert_model.encode(patterns)
        y = self.label_encoder.fit_transform(tags)

        # Compute mean embedding per intent
        unique_tags = list(self.label_encoder.classes_)
        tag_to_embeddings = {tag: [] for tag in unique_tags}

        for i, tag in enumerate(tags):
            tag_to_embeddings[tag].append(X[i])

        self.X = np.array([np.mean(embeddings, axis=0)
                           for embeddings in tag_to_embeddings.values()])
        self.y = np.array([self.label_encoder.transform([tag])[0]
                           for tag in tag_to_embeddings.keys()])

        self._save_resources()

    def teach_new_intent(self, pattern: str, response: str, tag: str):
        """Add new intent at runtime"""
        # Encode new pattern
        new_embedding = self.sbert_model.encode([pattern.lower()])[0]

        if tag not in self.label_encoder.classes_:
            # Handle new tag
            old_classes = list(self.label_encoder.classes_)
            old_classes.append(tag)
            self.label_encoder.fit(old_classes)

            # Initialize new embedding
            self.X = np.vstack([self.X, new_embedding])
            self.y = np.append(self.y, self.label_encoder.transform([tag])[0])
            self.responses_dict[tag] = [response]
        else:
            # Update existing intent
            tag_id = self.label_encoder.transform([tag])[0]
            idx = np.where(self.y == tag_id)[0][0]

            # Update embedding (average with existing)
            self.X[idx] = np.mean([self.X[idx], new_embedding], axis=0)
            self.responses_dict[tag].append(response)

        self._save_resources()

    def _save_resources(self):
        """Save all trained resources"""
        with open("data/sbert_embeddings.pkl", "wb") as f:
            pickle.dump(self.X, f)
        with open("data/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        with open("data/responses_dict.pkl", "wb") as f:
            pickle.dump(self.responses_dict, f)
        with open("data/y_labels.pkl", "wb") as f:
            pickle.dump(self.label_encoder.inverse_transform(self.y), f)


def initialize_knowledge_bases():
    """Initialize wiki and exercise knowledge bases"""

    def get_sentences_from_url(url, cache_path):
        CACHE = pathlib.Path(cache_path)
        if CACHE.exists():
            return json.loads(CACHE.read_text())

        article = Article(url)
        article.download()
        article.parse()
        sentences = nltk.sent_tokenize(article.text)
        CACHE.write_text(json.dumps(sentences, ensure_ascii=False, indent=2))
        return sentences

    # Initialize SBERT model
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load fitness knowledge base
    fitness_url = "https://en.wikipedia.org/wiki/Strength_training"
    wiki_sentences = get_sentences_from_url(fitness_url, "data/fitness_wiki.json")
    wiki_embeddings = sbert_model.encode(wiki_sentences)

    # Load exercises
    with open("data/exercises.json", "r") as f:
        exercises = json.load(f)["exercises"]

    exercise_sentences = []
    exercise_lookup = []

    for ex in exercises:
        sentence = f"{ex['name']} targets {', '.join(ex['primaryMuscles'])}. Category: {ex['category']}."
        exercise_sentences.append(sentence)
        exercise_lookup.append(ex)

    exercise_embeddings = sbert_model.encode(exercise_sentences)

    # Save all knowledge bases
    with open("data/wiki_sentences.json", "w") as f:
        json.dump(wiki_sentences, f, ensure_ascii=False, indent=2)
    with open("data/wiki_embeddings.pkl", "wb") as f:
        pickle.dump(wiki_embeddings, f)
    with open("data/exercise_sentences.json", "w") as f:
        json.dump(exercise_sentences, f, ensure_ascii=False, indent=2)
    with open("data/exercise_embeddings.pkl", "wb") as f:
        pickle.dump(exercise_embeddings, f)
    with open("data/exercise_lookup.pkl", "wb") as f:
        pickle.dump(exercise_lookup, f)


if __name__ == "__main__":
    # Initialize main intent classifier
    trainer = ChatbotTrainer()
    trainer.initialize_from_json("data/data.json")

    # Initialize knowledge bases
    initialize_knowledge_bases()

    print("AI Training completed")
