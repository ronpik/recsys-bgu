import sys
import time

from scripts.cfmodel_runner import evaluate

sys.path.append('../recsys')
from recsys.cf import RecommenderSystem
import pandas as pd
from recsys.utils.data.yelp_dataset import load_yelp_dataset, split_dataset
from recsys.cf.sentiment import generate_feature_set


if __name__ == "__main__":
    train_path = "/home/ron/data/studies/bgu/recsys/ex1/data/trainData.csv" #sys.argv[1]
    test_path = "/home/ron/data/studies/bgu/recsys/ex1/data/testData.csv" #sys.argv[2]
    train_feature_set_path = "/home/ron/data/studies/bgu/recsys/ex1/data/yelp_train_feature_set.pkl" #sys.argv[3]
    test_feature_set_path = "/home/ron/data/studies/bgu/recsys/ex1/data/yelp_test_feature_set.pkl" #sys.argv[4]

    start = time.time()
    print(f"load train data: {train_path}")
    train_df = load_yelp_dataset(train_path, use_text=True)
    print(f"load test data: {train_path}")
    test_df = load_yelp_dataset(test_path, use_text=True)
    end = time.time()
    print(f"loading data took {end - start:.2f}s")

    print(f"train data loaded with shape: {train_df.shape}")
    print(f"test data loaded with shape: {test_df.shape}")

    y_train = test_df.stars
    X_train, X_test = None, None
    if bool(train_feature_set_path) and bool(test_feature_set_path):
        print("load pre-calculated features")
        X_train = pd.read_pickle(train_feature_set_path)
        y_train = train_df.stars 

        X_test = pd.read_pickle(test_feature_set_path)
        y_test = test_df.stars
    else:
        X_train = generate_feature_set(train_df.text)
        y_train = train_df.stars 
        X_test = generate_feature_set(train_df.text)
        y_test = test_df.stars

    recsys_model = RecommenderSystem()
    recsys_model.TrainSentimentModel(X_train, y_train)
    y_pred = recsys_model.PredictRating(X_test, "sentiment")
    users = test_df.user_id
    y_true = test_df.stars
    evaluate(y_true, y_pred, users)

        











