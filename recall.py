import torch
import torch.nn as nn
import faiss
import json
from tqdm import tqdm
import numpy as np

# unique_actors, language_encoder, ratings, train_dataset, df = get_data()
# num_users = ratings['userId'].max() + 1
# num_items = ratings['movieId'].max() + 1
# num_actors = len(unique_actors) + 1 # 176047 + 1
# num_langs = len(language_encoder.tokens.keys()) + 1 # 88 + 1
# all_movieIds = ratings['movieId'].unique()

# model = NCF(num_users, num_items, train_dataset, df, all_movieIds, num_actors, num_langs)
# # model = NCF()
# model.load_state_dict(torch.load('./checkpoint/model.pth'))


model = torch.load("./checkpoint/model_small.pt")

user_embedding = model.user_embedding
movie_embedding = model.item_embedding
actor_embedding = model.cast_embedding
language_embedding = model.language_embedding

del model

print(type(user_embedding))
print(user_embedding.weight.shape)
print(type(movie_embedding))
print(movie_embedding.weight.shape)
print(type(actor_embedding))
print(actor_embedding.weight.shape)
print(type(language_embedding))
print(language_embedding.weight.shape)
# exit()
user_embedding = user_embedding.weight.detach().numpy()
movie_embedding = movie_embedding.weight.detach().numpy()


def index_train(embedding):

    d = embedding.shape[-1]
    nlist = 50  #  聚类中心个数
    # k = 10      # 查找最相似的k个向量
    quantizer = faiss.IndexFlatL2(d)  # 量化器
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        # METRIC_L2计算L2距离, 或faiss.METRIC_INNER_PRODUCT计算内积
    assert not index.is_trained   #倒排表索引类型需要训练
    index.train(embedding)  # 训练数据集应该与数据库数据集同分布
    assert index.is_trained
    # index.nprobe :查找聚类中心的个数 默认为1 
    index.nprobe = 50 # default nprobe is 1, try a few more
    index.add(embedding)
    # index.nprobe = 50  # 选择n个维诺空间进行索引,
    # dis, ind = index.search(query, k)
    return index


def get_user2movie_data():
    user2movie = {}
    with open("./output/ratings.csv", mode="r") as file:
        for line in tqdm(file.readlines()[1:],desc="get_user2movie_data"):
            line = line.split(",")
            user = line[0]
            movie = line[1]
            rating = float(line[2])
            if rating>4.0:
                if user not in user2movie.keys():
                    user2movie[user] = []
                user2movie[user].append(int(movie))
    with open("./output/user2movie.json", mode="w") as file:
        json.dump(user2movie,file)


get_user2movie_data()

user_index = index_train(user_embedding)
print(user_index)

query = user_embedding[:1]
print(query)
dis, ind = user_index.search(query,100)

print(dis)
print(ind)

user_sim = user_embedding[ind[0]]
print(user_sim.shape)

with open("./output/user2movie.json", mode="r") as file:
    user2movie = json.load(file)

movies_sim = []
for id in ind[0]:
    movies = user2movie.get(str(id),[])
    movies_sim.extend(movies)

print(movies_sim)
print(len(movies_sim))
