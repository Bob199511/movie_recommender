from fastapi import FastAPI
import json
import torch
import numpy as np

from recall import Recaller

app = FastAPI()

with open("./config.json", mode="r") as file:
    config = json.load(file)

with open("./output/meta_data.json", mode="r") as file:
    meta = json.load(file)

model = torch.load(config['model_path'])
recaller = Recaller(config['model_path'], "./index", True, True)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/recommend/{user_id}")
def recommend(user_id:int):
    recall_movies = recaller.get_movies_for_user(user_id)
    input_data = []
    for movie_id in recall_movies:
        if str(movie_id) not in meta.keys():
            print(str(movie_id)+"Not exist!")
            continue
        input = [user_id, movie_id] + meta[str(movie_id)]
        input_data.append(input)
    input_data = np.array(input_data).astype(np.float32)
    print(input_data.shape)
    result = np.squeeze(model(torch.tensor(input_data)).detach().numpy())
    K = 10
    topK_movies = [j for j in np.argsort(result)[::-1][0:K].tolist()]
    print(result)
    print(result.shape)
    print(topK_movies)
    print(len(topK_movies))
    return {"user_id":user_id,
            "topK_movies":topK_movies}