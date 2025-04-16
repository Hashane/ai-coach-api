import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

from app.chatbot.utils import mean_pooling
from app.db.models import User
from sqlalchemy.orm import Session
from transformers import AutoTokenizer, AutoModel
import torch

os.chdir('app/chatbot')

# Load trained SBERT embeddings, label encoder, and responses
with open("data/sbert_embeddings.pkl", "rb") as f:
    X = pickle.load(f)

with open("data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("data/responses_dict.pkl", "rb") as f:
    responses_dict = pickle.load(f)

# Load SBERT model
#sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


# Load tokenizer and transformer model
tokenizer = AutoTokenizer.from_pretrained('jgammack/distilbert-base-mean-pooling')
model = AutoModel.from_pretrained('jgammack/distilbert-base-mean-pooling')



def get_similar_response(user_input, user: User, db: Session):
    # Tokenize and get model output
    inputs = tokenizer([user_input.lower()], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    embedding = mean_pooling(model_output, inputs['attention_mask'])

    # Normalize like SBERT would
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    # Convert to numpy for cosine similarity
    input_embedding = embedding.cpu().numpy()

    similarities = cosine_similarity(input_embedding, X)[0]

    # Find best match
    best_match_index = np.argmax(similarities)
    confidence = similarities[best_match_index]
    predicted_label = label_encoder.classes_[best_match_index]

    if confidence > 0.3:
        if predicted_label == "bmi":
            return random.choice(responses_dict[predicted_label])
        elif predicted_label == "workout_plan":
            return "plan bitch."
        else:
            return random.choice(responses_dict[predicted_label])
    else:
        return "I'm not sure what you mean. Can you rephrase?"


# def get_user_bmi(user: User, db: Session):
#     user_data = db.query(User).filter(User.id == user.id).first()
#     if user_data:
#         weight = user_data.weight
#         height = user_data.height
#         bmi = weight / (height ** 2)
#         return round(bmi, 2)
#     else:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User data not found")
