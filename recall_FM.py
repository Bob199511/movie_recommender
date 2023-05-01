import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from surprise import Dataset, Reader

# 加载数据
df = pd.read_csv('./data/ratings_small.csv')
# data = Dataset.load_builtin('ml-100k')
reader = Reader(rating_scale=(1, 5))
# data = data.build_full_trainset().build_testset()
# df = pd.DataFrame(data, columns=["user_id", "item_id", "rating"])

# 划分数据集
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
# Read in data
def loadData(dataset):
    data = []
    y = []
    users=set()
    items=set()
    for sample in dataset.iterrows():
        (user,movieid,rating,ts)=sample[1]['userId'], sample[1]['movieId'], sample[1]['rating'], sample[1]['timestamp']
        data.append({ "user_id": str(user), "movie_id": str(movieid)})
        y.append(float(rating))
        users.add(user)
        items.add(movieid)

    return (data, np.array(y), users, items)
(train_data1, y_train, train_users, train_items) = loadData(train_data)
(test_data1, y_test, test_users, test_items) = loadData(test_data)

v = DictVectorizer()
X_train = v.fit_transform(train_data1)
X_test = v.transform(test_data1)

# 训练 FM 模型
fm = pylibfm.FM(num_factors=10, num_iter=10, verbose=True, task="classification", initial_learning_rate=0.001, learning_rate_schedule="optimal")
fm.fit(X_train,y_train)

# 使用 FM 模型进行召回推荐
users = test_data["userId"].unique()
items = df["movieId"].unique()
# X_test = v.transform([{"userId": u, "movieId": i} for u in users for i in items])
X_test = v.transform([{"userId": 1, "movieId": i} for i in items])
y_pred = fm.predict(X_test)

print(y_pred)
# 整理召回结果
recall = pd.DataFrame({
    "user_id": np.repeat(4, len(items)),
    "movie_id": items,
    "score": y_pred
})

# 按用户分组，取 Top-N 推荐结果
N = 1000
top_n = recall.groupby("user_id").apply(lambda x: x.nlargest(N, "score")).reset_index(drop=True)
top_n["rank"] = top_n.groupby("user_id").cumcount() + 1

# 输出召回结果
print(top_n)
