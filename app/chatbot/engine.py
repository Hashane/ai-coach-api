import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

from app.chatbot.utils import mean_pooling, extract_user_facts, extract_user_preferences
from app.db.crud import get_user_facts, save_user_facts, save_user_preferences, save_message, get_last_bot_message
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

# Load tokenizer and transformer model
tokenizer = AutoTokenizer.from_pretrained('jgammack/distilbert-base-mean-pooling')
model = AutoModel.from_pretrained('jgammack/distilbert-base-mean-pooling')


def get_similar_response(user_input, user: User, db: Session):
    save_message(user.id, user_input, is_bot=False, db=db)

    facts = extract_user_facts(user_input)
    preferences = extract_user_preferences(user_input)
    save_user_preferences(user.id, preferences, db)
    save_user_facts(user.id, facts, db)

    # Tokenize and get model output
    inputs = tokenizer([user_input.lower()], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    embedding = mean_pooling(model_output, inputs['attention_mask'])

    # Normalize
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
            user_facts = get_user_facts(user.id, db)
            response = f"Since your goal is to {user_facts['goal']}, I recommend a high-protein diet and strength training."
        elif predicted_label == "workout_plan":
            response = "plan bitch."
        elif predicted_label == "agree":
            last_bot_message = get_last_bot_message(user.id, db)
            response = "ok"
            if last_bot_message and "meal" in last_bot_message.lower():
                response = "Here’s a sample meal plan tailored for your goal. Let me know if you want a vegetarian or high-protein version!"
                save_message(user.id, response, is_bot=True, db=db)
        else:
            response = random.choice(responses_dict[predicted_label])
    else:
        response = "I'm not sure what you mean. Can you rephrase?"

    save_message(user.id, response, is_bot=True, db=db)
    return response

# def get_user_bmi(user: User, db: Session):
#     user_data = db.query(User).filter(User.id == user.id).first()
#     if user_data:
#         weight = user_data.weight
#         height = user_data.height
#         bmi = weight / (height ** 2)
#         return round(bmi, 2)
#     else:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User data not found")
