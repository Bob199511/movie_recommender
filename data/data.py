import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import re

def get_data():
    ratings = pd.read_csv('./data/ratings.csv',
                          parse_dates=['timestamp'])
    # sample 1% of users
    rand_userIds = np.random.choice(ratings['userId'].unique(),
                            size=int(len(ratings['userId'].unique())*0.1),
                            replace=False)

    ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]
    print('There are {} rows of data from {} users'.format(len(ratings), len(rand_userIds)))
    ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'] \
        .rank(method='first', ascending=False)

    movies_metadata = pd.read_csv('./data/movies_metadata.csv')
    # Select columns of interest
    movies_metadata = movies_metadata[['id', 'adult', 'budget', 'genres', 'original_language', 'revenue']]
    # extract movie genre into list for each record
    movies_metadata['genres'] = movies_metadata['genres'].apply(lambda x: re.findall("'name': '([a-zA-Z ]*)'", x))

    credits = pd.read_csv('./data/credits.csv')
    # extract cast names into list for each record
    credits['cast'] = credits['cast'].apply(lambda x: re.findall("'name': '([a-zA-Z ]*)'", x))

    # Convert the column to numeric
    movies_metadata['id'] = movies_metadata['id'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    movies_metadata = movies_metadata[movies_metadata['id'].notna()]
    movies_metadata = movies_metadata.dropna()
    movies_metadata['id'] = movies_metadata['id'].astype(int)
    df = pd.merge(movies_metadata, credits,
                  how='inner',
                  on='id')
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(sparse_output=True)

    df = df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(df.pop('genres')),
            index=df.index,
            columns=mlb.classes_))
    # Binarise adult column
    df['adult'] = df['adult'].map({'False': 0, 'True': 1})
    from torchnlp.encoders import LabelEncoder
    # Encode Cast- we select first 5 actors from cast list
    df = df[df["cast"].str.len() != 0]
    actors = [a for b in df["cast"].tolist() for a in b]
    unique_actors = list(set(actors))
    encoder = LabelEncoder(unique_actors, reserved_labels=['unknown'], unknown_index=0)

    def encode_cast(x):
        x = encoder.batch_encode(x)
        x = x[:5]
        if len(x) != 5:
            m = nn.ConstantPad1d((0, 5 - len(x)), 0)
            x = m(x)
        x = x.numpy().astype(int)
        return x

    df['cast_sl'] = df['cast'].apply(encode_cast)
    df = df[['id', 'adult', 'budget', 'original_language', 'revenue', 'cast_sl',
             'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
             'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music',
             'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War',
             'Western']]
    language_encoder = LabelEncoder(df['original_language'].unique().tolist(), reserved_labels=['unknown'],
                                    unknown_index=0)

    df['original_language'] = df['original_language'].apply(lambda x: language_encoder.encode(x).numpy())
    links_table = pd.read_csv('./data/links.csv')
    links_table = links_table[['movieId', 'tmdbId']]
    df = pd.merge(df, links_table,
                  how='inner',
                  left_on='id',
                  right_on='tmdbId')
    df = df.drop(['tmdbId'], axis=1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[['budget', 'revenue']])
    df[['budget', 'revenue']] = scaler.transform(df[['budget', 'revenue']])
    df[['cast1', 'cast2', 'cast3', 'cast4', 'cast5']] = pd.DataFrame(df.cast_sl.tolist(), index=df.index)
    df = df.drop(['cast_sl'], axis=1)
    ratings = ratings.drop(['timestamp'], axis=1)
    # the last movie user has seen is rank 1.
    # Move that to test set
    # Train test split
    train_dataset = ratings[ratings['rank_latest'] != 1]
    test_dataset = ratings[ratings['rank_latest'] == 1]
    train_dataset.loc[:, 'rating'] = 1
    train_dataset = train_dataset.drop(['rank_latest'], axis=1)
    del movies_metadata
    del credits
    del links_table
    print("train_dataset.shape:", train_dataset.shape)
    train_dataset = train_dataset.reset_index(drop=True)
    print("train_dataset.head()", train_dataset.head())
    print("df.columns: ", df.columns)
    ratings.to_csv("./output/ratings.csv")
    train_dataset.to_csv("./output/train_dataset.csv")
    test_dataset.to_csv("./output/test_datset.csv")
    df.to_csv("./output/df.csv")


    return unique_actors, language_encoder, ratings, train_dataset, df


class MovieLensTrainDataset(Dataset):
    """MovieLens PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the movie ratings
        all_movieIds (list): List containing all movieIds

    """

    def __init__(self, ratings, metadata, all_movieIds):
        #         self.userId, self.movieId, self.cast, self.misc, self.language, self.res = self.get_dataset(ratings, metadata, all_movieIds)
        self.data, self.res = self.get_dataset(ratings, metadata, all_movieIds)

    def __len__(self):
        return len(self.res)

    def __getitem__(self, idx):
        return self.data[idx], self.res[idx]

    def get_dataset(self, ratings, metadata, all_movieIds):
        #         print("PRINT 3", psutil.virtual_memory().percent)
        new_samples = []
        user_item_set = set(zip(ratings['userId'], ratings['movieId']))

        num_negatives = 4
        for u, i in tqdm(user_item_set):
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_movieIds)
                new_samples.append([u, negative_item, 0])

        # this df now caontains all negative samples appended to the positive samples
        ratings = ratings.append(pd.DataFrame(new_samples, columns=ratings.columns))

        # combine ratings with movies_metadata
        ratings = pd.merge(ratings, metadata,
                           how='inner',
                           on='movieId').drop(['id'], axis=1)

        ratings = ratings[['userId', 'movieId', 'original_language', 'adult', 'budget',
                           'revenue', 'Action', 'Adventure', 'Animation', 'Comedy',
                           'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign',
                           'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
                           'TV Movie', 'Thriller', 'War', 'Western', 'cast1', 'cast2', 'cast3', 'cast4', 'cast5',
                           'rating']]
        ratings = ratings.dropna()

        #         print("PRINT 4", psutil.virtual_memory().percent)
        return ratings[ratings.columns[:-1]].to_numpy().astype(np.float32), ratings['rating'].to_numpy().astype(int)