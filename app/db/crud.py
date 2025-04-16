from sqlalchemy.orm import Session
from app.db.models import UserFact, UserPreferences


def save_user_facts(user_id: int, facts: dict, db: Session):
    for key, value in facts.items():
        existing = db.query(UserFact).filter_by(user_id=user_id, key=key).first()
        if existing:
            existing.value = str(value)
        else:
            new_fact = UserFact(user_id=user_id, key=key, value=str(value))
            db.add(new_fact)
    db.commit()


def save_user_preferences(user_id: int, facts: list, db: Session):
    for fact in facts:
        value = fact["value"]
        sentiment = fact["sentiment"]
        category = fact["category"]

        existing = db.query(UserPreferences).filter_by(user_id=user_id, category=category, value=value).first()

        if existing:
            # If the preference already exists, update the sentiment
            existing.sentiment = sentiment
        else:
            new_preference = UserPreferences(
                user_id=user_id,
                category=category,
                value=value,
                sentiment=sentiment
            )
            db.add(new_preference)

    db.commit()


def get_user_facts(user_id: int, db: Session):
    facts = db.query(UserFact).filter_by(user_id=user_id).all()
    return {fact.key: fact.value for fact in facts}
