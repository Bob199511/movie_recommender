import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from model import NCF

test_dataset = pd.read_csv('./output/test_dataset.csv').drop(labels='Unnamed: 0', axis=1)
print(test_dataset.columns)
train_dataset = pd.read_csv('./output/train_dataset.csv').drop(labels='Unnamed: 0', axis=1)
df = pd.read_csv('./output/df.csv').drop(labels='Unnamed: 0', axis=1)
print(test_dataset.head())
test_dataset.loc[:, 'rating'] = 1
# 预测用户最近看的一次电影
test_dataset = test_dataset.drop(['rank_latest', 'rating'], axis=1)
test_dataset = test_dataset.reset_index(drop=True)

ratings = pd.read_csv('./output/ratings.csv').drop(labels='Unnamed: 0', axis=1)
# num_users = ratings['userId'].max() + 1
# num_items = ratings['movieId'].max() + 1
# num_actors = 176047 + 1 #len(unique_actors) + 1 #
# num_langs = 88 + 1 #len(language_encoder.tokens.keys()) + 1 #
all_movieIds = ratings['movieId'].unique()
# model = NCF(num_users, num_items, train_dataset, df, all_movieIds, num_actors, num_langs)
# model.load_state_dict(torch.load('./checkpoint/model.pth'))
model_path = "./checkpoint/model_small.pt"
model = torch.load(model_path)
print("Loading from ", model_path)

# torch.save(model.state_dict(), './checkpoint/model_small.pth')
# model = NCF(train_dataset, df, all_movieIds)
model.ratings = ratings
model.eval()

def test_model():
    # User-item pairs for testing
    test_user_item_set = set(zip(test_dataset['userId'], test_dataset['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()

    k = 10

    unique_movies_covered = []
    num_users_to_check_for = 1000
    user_counter = 0

    hits = []
    hits5 = []
    for (u, i) in tqdm(test_user_item_set):
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_movieIds) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 19))
        test_items = selected_not_interacted + [i]  # [100 movies]
        user = [u] * 20
        user_movie = list(zip(user, test_items))
        user_movie = pd.DataFrame(user_movie, columns=test_dataset.columns)
        merged_df = pd.merge(user_movie, df,
                             how='inner',
                             on='movieId').drop(['id'], axis=1)
        merged_df = merged_df[['userId', 'movieId', 'original_language', 'adult', 'budget',
                               'revenue', 'Action', 'Adventure', 'Animation', 'Comedy',
                               'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign',
                               'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
                               'TV Movie', 'Thriller', 'War', 'Western', 'cast1', 'cast2', 'cast3', 'cast4', 'cast5']]

        test_items = merged_df['movieId'].tolist()
        predicted_labels = np.squeeze(model(torch.tensor(merged_df.to_numpy().astype(np.float32))).detach().numpy())

        top10_items = [test_items[j] for j in np.argsort(predicted_labels)[::-1][0:k].tolist()]

        if user_counter < 1000:
            unique_movies_covered += top10_items
            user_counter += 1

        top5_items = [test_items[j] for j in np.argsort(predicted_labels)[::-1][0:5].tolist()]

        if i in top10_items:
            hits.append(1)
        elif i not in test_items:
            pass
        else:
            hits.append(0)

        if i in top5_items:
            hits5.append(1)
        elif i not in test_items:
            pass
        else:
            hits5.append(0)

    print("The Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))

    print("The Hit Ratio @ 5 is {:.2f}".format(np.average(hits5)))

def run():
    pass
test_model() # 队训练好的模型进行验证
run() # 实时运行模型