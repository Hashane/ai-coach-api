from fastapi import FastAPI
from app.routes import auth

app = FastAPI()

app.include_router(auth.router)
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
