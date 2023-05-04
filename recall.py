import torch
import torch.nn as nn
import faiss
import json
from tqdm import tqdm
import numpy as np
import os

# unique_actors, language_encoder, ratings, train_dataset, df = get_data()
# num_users = ratings['userId'].max() + 1
# num_items = ratings['movieId'].max() + 1
# num_actors = len(unique_actors) + 1 # 176047 + 1
# num_langs = len(language_encoder.tokens.keys()) + 1 # 88 + 1
# all_movieIds = ratings['movieId'].unique()

# model = NCF(num_users, num_items, train_dataset, df, all_movieIds, num_actors, num_langs)
# # model = NCF()
# model.load_state_dict(torch.load('./checkpoint/model.pth'))


# model = torch.load("./checkpoint/model_small.pt")

# user_embedding = model.user_embedding
# movie_embedding = model.item_embedding
# actor_embedding = model.cast_embedding
# language_embedding = model.language_embedding

# del model

# print(type(user_embedding))
# print(user_embedding.weight.shape)
# print(type(movie_embedding))
# print(movie_embedding.weight.shape)
# print(type(actor_embedding))
# print(actor_embedding.weight.shape)
# print(type(language_embedding))
# print(language_embedding.weight.shape)
# # exit()
# user_embedding = user_embedding.weight.detach().numpy()
# movie_embedding = movie_embedding.weight.detach().numpy()


class Recaller(object):
    def __init__(self, emb_path, index_path, index_train=False, dict_reload=False) -> None:
        model = torch.load(emb_path)
        self.user_embedding = model.user_embedding.weight.detach().numpy()
        self.movie_embedding = model.item_embedding.weight.detach().numpy()

        self.topK = 100
        self.usercf_k = 20
        self.usercf_n = 300
        self.moviecf_k = 20
        self.moviecf_n = 200
        
        if not os.path.exists(os.path.join(index_path,"user.index")) or index_train:
            self.user_index = self.index_train(self.user_embedding)
            faiss.write_index(self.user_index, os.path.join(index_path,"user.index"))
        else:
            self.user_index = faiss.read_index(os.path.join(index_path,"user.index"))
        
        if not os.path.exists(os.path.join(index_path,"movie.index")) or index_train:
            self.movie_index = self.index_train(self.movie_embedding)
            faiss.write_index(self.movie_index, os.path.join(index_path,"movie.index"))
        else:
            self.movie_index = faiss.read_index(os.path.join(index_path,"movie.index"))
        

        if not os.path.exists(os.path.join(index_path,"user2movie.json")) or not os.path.exists(os.path.join(index_path,"topKmovies.json")) or dict_reload:
            self.user2movie, self.topKmovies = self.get_user2movie_data(index_path)
        else:
            with open(os.path.join(index_path,"user2movie.json"), mode="r") as file:
                self.user2movie = json.load(file)
            with open(os.path.join(index_path,"topKmovies.json"), mode="r") as file:
                self.topKmovies = json.load(file)          

    def index_train(self, embedding):

        d = embedding.shape[-1]
        nlist = 1  #  聚类中心个数
        # k = 10      # 查找最相似的k个向量
        quantizer = faiss.IndexFlatL2(d)  # 量化器
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            # METRIC_L2计算L2距离, 或faiss.METRIC_INNER_PRODUCT计算内积
        assert not index.is_trained   #倒排表索引类型需要训练
        index.train(embedding)  # 训练数据集应该与数据库数据集同分布
        assert index.is_trained
        # index.nprobe :查找聚类中心的个数 默认为1 
        index.nprobe = 1 # default nprobe is 1, try a few more
        index.add(embedding)
        # index.nprobe = 50  # 选择n个维诺空间进行索引,
        # dis, ind = index.search(query, k)
        return index

    def get_user2movie_data(self, dir):
        user2movie = {}
        movie_count = {}
        movie_rating = {}
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
                movie_count[movie] = movie_count.get(movie, 0) + 1
                movie_rating[movie] = movie_rating.get(movie, 0) + rating
        with open(os.path.join(dir,"user2movie.json"), mode="w") as file:
            json.dump(user2movie,file)
        for movie,rating in movie_rating.items():
            movie_rating[movie] = rating/movie_count[movie]
        movie_sort = sorted(list(movie_count.items()), key=lambda x:x[1], reverse=True)
        movie_top = []
        for movie,count in movie_sort:
            if movie_rating[movie]>=4.0:
                movie_top.append(int(movie))
                if len(movie_top)>=self.topK:
                    break
        with open(os.path.join(dir,"topKmovies.json"), mode="w") as file:
            json.dump(movie_top,file)
        return user2movie, movie_top
    
    def userCF(self, user_id, k, n):
        query = self.user_embedding[user_id:user_id+1]
        dis, ind = self.user_index.search(query, k)

        movies_sim = []
        for id in ind[0]:
            movies = self.user2movie.get(str(id),[])
            movies_sim.extend(movies)
            if len(movies_sim)>=n:
                movies_sim = movies_sim[:n]
                break
        return movies_sim
    
    def movieCF(self, user_id, k, n):
        movie_ids = self.user2movie.get(str(user_id), [])
        if not movie_ids:
            return []
        query = self.movie_embedding[movie_ids]
        dis, ind = self.movie_index.search(query, k)
        movies_sim = []
        for t in range(k):
            for i in range(len(query)):
                movies_sim.append(ind[i][t])
                if len(movies_sim)>=n:
                    break
            if len(movies_sim)>=n:
                movies_sim = movies_sim[:n]
                break
        return movies_sim

    def get_movies_for_user(self, userid):
        user_sim = self.userCF(userid, self.usercf_k, self.usercf_n)
        movie_sim = self.movieCF(userid, self.moviecf_k, self.moviecf_n)
        return list(set(self.topKmovies + user_sim + movie_sim))


if __name__=="__main__":
    recaller = Recaller("./checkpoint/model_small.pt", "./index", True, True)
    movies = recaller.get_movies_for_user(220)
    print(movies)
    print(len(movies))