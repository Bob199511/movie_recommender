import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import re
from data_loader import get_data, MovieLensTrainDataset
from model import NCF

unique_actors, language_encoder, ratings, train_dataset, df = get_data()
# num_users = ratings['userId'].max() + 1
# num_users = len(ratings['userId'].unique()) + 1
# num_items = ratings['movieId'].max() + 1
# num_items = len(ratings['movieId'].unique()) + 1
# num_actors = len(unique_actors) + 1 # 176047 + 1
# num_langs = len(language_encoder.tokens.keys()) + 1 # 88 + 1
all_movieIds = ratings['movieId'].unique()

model = NCF(train_dataset, df, all_movieIds)
rec = dict()
# rec['num_users'] = num_users
# rec['num_items'] = num_items
# rec['num_actors'] = num_actors
# rec['num_langs'] = num_langs
rec['all_movieIds'] = all_movieIds
import json
with open('./output/record', 'w+') as f:
    f.write(str(model))
    for key in rec.items():
        f.write(str(key))
trainer = pl.Trainer(max_epochs=20, accelerator='gpu', devices=1, reload_dataloaders_every_n_epochs=1,
                     enable_progress_bar=True, logger=True, enable_checkpointing=True)

trainer.fit(model)
# torch.save(model.state_dict(), './checkpoint/model_2.pth')
torch.save(model, "./checkpoint/model_small.pt")