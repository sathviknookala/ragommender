from fastapi import FastAPI
from retrieval import hybrid_search
from retrieval import get_similar 
from retrieval import get_user_profile

app = FastAPI()

@app.get("/v1/user/{user_id}")
def user_profile(user_id: str):
    return get_user_profile(user_id)

@app.get("/v1/similar/{user_id}")
def similar(user_id: str):
    return get_similar(user_id)

@app.post("/v1/reommend")
def recommend(user_id: str, k: int):
    return hybrid_search(user_id, k)