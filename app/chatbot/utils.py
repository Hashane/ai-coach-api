import string
from typing import List

import torch
import spacy
from transformers import pipeline
import random
import ast
from typing import Union, List

classifier = pipeline("zero-shot-classification")


def extract_user_preferences(message: str):
    doc = nlp(message)
    results = []

    noise_words = {"lot", "something", "things", "thing", "time", "do"}
    negative_phrases = ["don't like", "do not like", "hate", "dislike", "can't stand"]
    positive_phrases = ["like", "love", "prefer", "enjoy"]
    contrast_words = ["but", "however", "although", "except"]

    candidate_labels = [
        "strength exercise",
        "cardio exercise",
        "fitness class",
        "sports",
        "food",
        "other"
    ]

    def get_sentiment_clauses(sent):
        """Split sentence into clauses with consistent sentiment"""
        clauses = []
        current_clause = []
        current_sentiment = None

        for token in sent:
            # Detect sentiment change
            token_text = token.text.lower()

            if any(neg in token_text for neg in negative_phrases):
                if current_clause and current_sentiment != "dislike":
                    clauses.append((" ".join(current_clause), current_sentiment))
                    current_clause = []
                current_sentiment = "dislike"
            elif any(pos in token_text for pos in positive_phrases):
                if current_clause and current_sentiment != "like":
                    clauses.append((" ".join(current_clause), current_sentiment))
                    current_clause = []
                current_sentiment = "like"
            elif token.text.lower() in contrast_words:
                if current_clause:
                    clauses.append((" ".join(current_clause), current_sentiment))
                current_clause = []
                current_sentiment = None  # Reset for next clause

            current_clause.append(token.text)

        if current_clause:
            clauses.append((" ".join(current_clause), current_sentiment))

        return clauses

    def process_clause(text, sentiment):
        """Process individual clause"""
        if not sentiment:
            return []

        clause_doc = nlp(text)
        clause_results = []

        for chunk in clause_doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if (chunk.root.pos_ in ["NOUN", "PROPN"] and
                    chunk_text not in noise_words and
                    not any(phrase in chunk_text for phrase in positive_phrases + negative_phrases)):

                category = classifier(chunk_text, candidate_labels, multi_label=False)["labels"][0]

                if any(workout_term in category for workout_term in ["exercise", "equipment", "fitness"]):
                    clause_results.append({
                        "value": chunk_text,
                        "sentiment": sentiment,
                        "category": category,
                        "type": "workout"
                    })

        return clause_results

    for sent in doc.sents:
        clauses = get_sentiment_clauses(sent)
        for clause_text, sentiment in clauses:
            results.extend(process_clause(clause_text, sentiment))

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


def generate_title_from_message(message: str) -> str:
    return message[:40] + "..." if len(message) > 40 else message


def calculate_bmi(weight: float, height: float) -> float:
    return weight / (height ** 2)


def get_bmi_category(bmi: float) -> string:
    bmi_category = (
        "underweight" if bmi < 18.5 else
        "normal" if bmi < 25 else
        "overweight" if bmi < 30 else
        "obese"
    )
    return bmi_category


def get_recommendation(goal: str, bmi: float) -> str:
    recommendations = {
        "lose weight": {
            "underweight": "a balanced diet with strength training to build healthy muscle",
            "normal": "moderate calorie deficit with cardio and resistance training",
            "overweight": "a calorie-controlled diet with regular cardio exercise",
            "obese": "gradual weight loss through diet modification and low-impact exercise"
        },
        "gain muscle": {
            "underweight": "calorie surplus with strength training 3-5x/week",
            "normal": "slight calorie surplus with progressive overload training",
            "overweight": "body recomposition approach with strength training",
            "obese": "focus on strength training while maintaining slight calorie deficit"
        }
    }

    bmi_category = get_bmi_category(bmi)

    return recommendations.get(goal, {}).get(bmi_category, "regular exercise and balanced nutrition")


def generate_workout_plan(
        days: Union[str, List[str]],
        goal: str,
        experience: str = "beginner"
) -> str:
    """
    Generates a workout plan for dynamic days input
    - Handles both string and list inputs
    - Returns formatted plan
    """

    workout_days = []

    if isinstance(days, str):
        try:
            workout_days = ast.literal_eval(days)
            if not isinstance(workout_days, list):
                workout_days = days.split(',')
        except:
            workout_days = days.split(',')

    # Ensure we have a clean list of days
    workout_days = [day.strip(" []'\"") for day in workout_days if day.strip()]
    if not workout_days:
        workout_days = ["Monday", "Wednesday", "Friday"]  # Default

    # 2. EXERCISE SELECTION
    exercises = {
        "build muscle": {
            "beginner": ["Push-ups", "Dumbbell rows", "Bodyweight squats"],
            "intermediate": ["Bench press", "Deadlifts", "Pull-ups"],
            "advanced": ["Weighted pull-ups", "Barbell squats", "Overhead press"]
        },
        "lose fat": {
            "beginner": ["Jumping jacks", "Bodyweight circuits", "Walking lunges"],
            "intermediate": ["Kettlebell swings", "Box jumps", "Battle ropes"],
            "advanced": ["Sprint intervals", "Complex lifts", "HIIT circuits"]
        }
    }

    # Load appropriate exercises
    goal = goal.lower()
    workout_type = exercises.get(goal, exercises["build muscle"])
    daily_exercises = workout_type.get(experience, workout_type["beginner"])

    # 3. PLAN GENERATION
    plan = []
    for day in workout_days:
        selected = random.sample(daily_exercises, min(3, len(daily_exercises)))
        plan.append(f"{day.strip()}: {', '.join(selected)}")

    return f"\n \n Your {len(workout_days)}-day '{goal}' plan:\n \n" + "\n".join(plan)


def generate_meal_plan(dietary_restrictions: List[str], goal: str, meals_per_day: int = 3) -> str:
   return ""
