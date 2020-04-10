import sys
import time

sys.path.append('..')
from recsys.cf import RecommenderSystem
import pandas as pd
from recsys.utils.data.yelp_dataset import load_yelp_dataset, split_dataset

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    start = time.time()
    print(f"load train data: {train_path}")
    train_df = load_yelp_dataset(train_path, use_text=False)
    print(f"load test data: {train_path}")
    test_df = load_yelp_dataset(test_path, use_text=False)
    end = time.time()
    print(f"loading data took {end - start:.2f}s")

    print(f"train data loaded with shape: {train_df.shape}")
    print(f"test data loaded with shape: {test_df.shape}")

    SupriseModel = RecommenderSystem(train_df, test_df)
    SupriseModel.TrainSupriseModel()













