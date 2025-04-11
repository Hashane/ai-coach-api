from fastapi import FastAPI, Depends, HTTPException, status
from app.auth.dependencies import authenticate_user, create_access_token, get_current_user
from app.db.connection import SessionLocal
from app.auth.models import Token, User

app = FastAPI()


@app.post("/token", response_model=Token)
async def login_for_access_token(username: str, password: str):
    db = SessionLocal()
    user = authenticate_user(db, username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return Token(access_token=access_token, token_type="bearer")


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
