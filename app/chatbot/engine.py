import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import json

from app.chatbot.utils import extract_user_facts, extract_user_preferences, generate_workout_plan, \
    get_recommendation, calculate_bmi, get_bmi_category, request_for_data, check_if_user_data_exists
from app.db.crud import get_user_facts, save_user_facts, save_user_preferences, save_message, get_last_bot_message, \
    get_user_preferences, get_workout_preferences
from app.db.models import User
from sqlalchemy.orm import Session

os.chdir('app/chatbot')

# Load trained SBERT embeddings, label encoder, and responses
with open("data/sbert_embeddings.pkl", "rb") as f:
    X = pickle.load(f)

with open("data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("data/responses_dict.pkl", "rb") as f:
    responses_dict = pickle.load(f)

# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_similar_response(user_input, user: User, conversation_id: int, db: Session):
    save_message(user.id, conversation_id, user_input, is_bot=False, db=db)

    preferences = extract_user_preferences(user_input)
    save_user_preferences(user.id, preferences, db)

    facts = extract_user_facts(user_input)
    save_user_facts(user.id, facts, db)

    response = "Thanks! I've captured the following details you shared with me:"
    if facts:
        if 'weight' in facts:
            response += f"\n- Weight: {facts['weight']} kg"

        if 'height' in facts:
            response += f"\n- Height: {facts['height']} cm"

        if 'goal' in facts:
            response += f"\n- Goal: {facts['goal']}"

        if 'gym_days' in facts:
            response += f"\n- Gym Days: {', '.join(facts['gym_days'])}"

        save_message(user.id, conversation_id, response, is_bot=True, db=db)
        return response, conversation_id

    if preferences:
        cleaned_preferences = [pref['value'] for pref in preferences]
        response = f"I captured your preferences: {', '.join(cleaned_preferences)}"
        save_message(user.id, conversation_id, response, is_bot=True, db=db)
        return response, conversation_id

    # Retrieve stored user data
    user_facts = get_user_facts(user.id, db)
    workout_prefs = get_workout_preferences(user.id, db)

    # Generate context-aware response
    input_embedding = sbert_model.encode([user_input.lower()], convert_to_tensor=True).cpu().numpy()
    similarities = cosine_similarity(input_embedding, X)[0]

    # Find the best match
    best_match_index = np.argmax(similarities)
    confidence = similarities[best_match_index]
    predicted_label = label_encoder.classes_[best_match_index]

    response_templates = {
        "bmi": [
            "Based on your height of {height} and weight of {weight}, your BMI is {bmi} You are currently {category}. "
            "For your goal to {goal}, I'd recommend {recommendation}.",

            "I see you're {height} tall and weigh {weight}. With your goal to {goal}, "
            "you should focus on {recommendation}."
        ],
        "workout_plan": [
            "Since you prefer {workout_prefs} and hit the gym on {gym_days},\n"
            "here's your customized plan: {plan}",

            "Your ideal workout plan for {gym_days}, preferring {workout_prefs}:\n{plan}"
        ],
        "diet": [
            "For your {goal} goal, considering you prefer {diet_prefs}, "
            "try this: {diet_plan}",

            "Your personalized meal plan (aligned with your {diet_prefs} preferences):\n{meal_ideas}"
        ]
    }

    if confidence > 0.6:
        if predicted_label == "bmi":
            if check_if_user_data_exists(user_facts):
                bmi = calculate_bmi(float(user_facts['weight']), float(user_facts['height']))
                recommendation = get_recommendation(user_facts['goal'], bmi)
                category = get_bmi_category(bmi)
                response = random.choice(response_templates["bmi"]).format(
                    height=user_facts['height'],
                    weight=user_facts['weight'],
                    bmi=round(bmi, 1),
                    goal=user_facts['goal'],
                    recommendation=recommendation,
                    category=category
                )
            else:
                response = request_for_data()


        elif predicted_label == "workout_plan":

            liked_workouts = [
                pref['activity']
                for pref in workout_prefs
                if pref['preference'] == 'like'
            ]
            if check_if_user_data_exists(user_facts):
                plan = generate_workout_plan(
                    user_facts['gym_days'],
                    user_facts['goal']
                )

                response = random.choice(response_templates["workout_plan"]).format(
                    workout_prefs=", ".join(liked_workouts),
                    gym_days=user_facts['gym_days'],
                    plan=plan
                )
            else:
                response = request_for_data()

        # elif predicted_label == "agree":
        #     last_bot_message = get_last_bot_message(user.id, db)
        #     if last_bot_message and "meal" in last_bot_message.lower():
        #         meal_plan = generate_meal_plan(
        #             user_prefs.get('dietary_restrictions', []),
        #             user_facts['goal']
        #         )
        #         response = random.choice(response_templates["diet"]).format(
        #             goal=user_facts['goal'],
        #             diet_prefs=", ".join(user_prefs.get('dietary_restrictions', ["no restrictions"])),
        #             meal_ideas=meal_plan
        #         )
        #     else:
        #         response = "Great! What else can I help you with?"

        else:
            # Fallback to generic responses with personalization
            generic_response = random.choice(responses_dict[predicted_label])
            response = f"{user_facts.get('name', 'There')}, {generic_response.lower()}"

    else:
        # Try semantic search fallback if confidence is low
        with open("data/kb_sentences.json", "r") as f:
            kb_sentences = json.load(f)

        with open("data/kb_embeddings.pkl", "rb") as f:
            kb_embeddings = pickle.load(f)

        # Encode user input
        query_embedding = sbert_model.encode([user_input])[0].reshape(1, -1)
        similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
        top_idx = np.argmax(similarities)

        if similarities[top_idx] > 0.4:
            response = f"Here’s what I found: {kb_sentences[top_idx]}"
        else:
            response = "I'm not sure I understand. Could you please rephrase?"

        # else:
        #     response = "Sorry, Can you rephrase that please?"

    save_message(user.id, conversation_id, response, is_bot=True, db=db)
    return response, conversation_id
