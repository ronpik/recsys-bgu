import sys
import time

sys.path.append('/Users/shaimeital/code/thesis/recsys-bgu')
from recsys.cf import RecommenderSystem
import pandas as pd
from recsys.utils.data.yelp_dataset import load_yelp_dataset
from recsys.cf.sentiment import generate_feature_set


if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    train_feature_set_path = sys.argv[3]
    test_feature_set_path = sys.argv[4]

    start = time.time()
    print(f"load train data: {train_path}")
    train_df = load_yelp_dataset(train_path, use_text=True)
    print(f"load test data: {train_path}")
    test_df = load_yelp_dataset(test_path, use_text=True)
    end = time.time()
    print(f"loading data took {end - start}s")
    print(f"train data loaded with shape: {train_df.shape}")
    print(f"test data loaded with shape: {test_df.shape}")

    if train_feature_set_path & test_feature_set_path:
        X_train = pd.read_pickle(train_feature_set_path)
        y_train = train_df.stars 

        X_test = pd.read_pickle(test_feature_set_path)
        y_test = test_df.stars

        SentimentModel = RecommenderSystem(X_train, X_test)
        SentimentModel.TrainSentimentModel()
    
    else:
        X_train = generate_feature_set(train_df.text)
        y_train = train_df.stars 
        X_test = generate_feature_set(train_df.text)
        y_test = test_df.stars

        SentimentModel = RecommenderSystem(X_train, X_test)
        SentimentModel.TrainSentimentModel()
        











