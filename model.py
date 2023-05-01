import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import re
from data.data import get_data, MovieLensTrainDataset

class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)

        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
            userId = data[0]
            movieId = data[1]
            misc = data[2:6]
            cast = data[6][:5]
            genre = data[7:]
    """

    def __init__(self, num_users, num_items, ratings, metadata, all_movieIds, num_actors, num_languages):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=16)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=16)

        self.misc_stack = nn.Sequential(
            nn.Linear(23, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.cast_embedding = nn.Embedding(num_embeddings=num_actors, embedding_dim=8)

        self.cast_stack = nn.Sequential(
            nn.Linear(8 * 5, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.language_embedding = nn.Embedding(num_embeddings=num_languages, embedding_dim=4)

        self.combiner = nn.Sequential(
            nn.Linear(32 * 2 + 4, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        #         self.fc1 = nn.Linear(in_features=32, out_features=64)
        self.fc1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128, eps=1e-12),
            nn.Dropout(0.1)
        )
        #         self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.LayerNorm(32, eps=1e-12),
            nn.Dropout(0.1)
        )
        self.output = nn.Linear(in_features=32, out_features=1)

        self.ratings = ratings
        self.all_movieIds = all_movieIds
        self.metadata = metadata

    def forward(self, data):
        # Pass through embedding layers
        user_embedded = self.user_embedding(data[:, 0].int())
        item_embedded = self.item_embedding(data[:, 1].int())
        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        misc_out = self.misc_stack(data[:, 3:-5])

        cast_emb_list = []
        for i in range(5):
            cast_emb_list.append(self.cast_embedding(data[:, -5 + i].int()))

        final_cast_emb = torch.cat(cast_emb_list, dim=-1)

        final_cast = self.cast_stack(final_cast_emb)

        lang_out = self.language_embedding(data[:, 2].int())

        combined = self.combiner(torch.cat([misc_out, final_cast, lang_out], dim=-1))

        # Pass through dense layer
        #         vector = nn.ReLU()(self.fc1(combined))
        vector = self.fc1(torch.cat([vector, combined], dim=-1))
        #         vector = nn.ReLU()(self.fc2(vector))
        vector = self.fc2(vector)

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        data, labels = batch
        predicted_labels = self(data)  # self calls 'forward()'
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MovieLensTrainDataset(self.ratings, self.metadata, self.all_movieIds),
                          batch_size=512, shuffle=True, num_workers=0)