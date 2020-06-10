from collections import Counter
import pandas as pd
import logging
import os

## Data

train_path = "part-00002.csv"

train_data = pd.read_csv(train_path)

train_data.head()

len(train_data.columns)

import sys

sys.path.append("/home/ron/workspace/recsys-bgu")

from kaggle.preprocess.features_hashing import FeaturesHashingProcessor

features_processor = FeaturesHashingProcessor()
features_processor.fit(train_data)
len(features_processor.features_names)

features = features_processor.transform(train_data)
print(features.shape)
features
