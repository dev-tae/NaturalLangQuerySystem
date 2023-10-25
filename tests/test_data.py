import sqlite3

import pandas as pd;

# df = pd.read_json('../data/spider/dev.json')
train_df = pd.read_json('../data/spider/train_spider.json')
dev_df = pd.read_json('../data/spider/dev.json')

train_df.head()
train_df.info()

dev_df.head()
dev_df.info()