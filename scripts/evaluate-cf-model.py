import random
import sys
import os

import numpy as np

from recsys.cf.advancedmodel import load_advanced_svd_model

module_dir = os.path.dirname(os.path.dirname(__file__))
print(module_dir)
sys.path.append(module_dir)

import time

from recsys.utils.data.yelp_dataset import load_yelp_dataset, reindex_data
from recsys.cf import RecommenderSystem, SVDModelEngine
from recsys.cf.basemodel import load_svd_model
from recsys.eval.evaltools import rmse


if __name__ == "__main__":
    # sys.argv[1]
    train_path = "/home/ron/data/studies/bgu/recsys/ex1/data/trainData.csv"
    # sys.argv[2]
    test_path = "/home/ron/data/studies/bgu/recsys/ex1/data/testData.csv"
    model_name = "svd++" #sys.argv[3]
    # model_path = "/home/ron/data/studies/bgu/recsys/ex1/data/base-svd-model.npz" # sys.argv[4]
    model_path = "/home/ron/data/studies/bgu/recsys/ex1/data/advanced-svd-model.npz" # sys.argv[4]
    start = time.time()
    print(f"load train data: {train_path}")
    train_df = load_yelp_dataset(train_path)
    print(f"load test data: {train_path}")
    test_df = load_yelp_dataset(test_path)
    end = time.time()
    print(f"loading data took {end - start:.2f} sec")
    print()
    print(f"train data loaded with shape: {train_df.shape}")
    print(f"test data loaded with shape: {test_df.shape}")
    print()
    print("re-indexing data")
    start = time.time()
    train_df, test_df = reindex_data(train_df, test_df)
    end = time.time()
    print(f"re-indexing took {end - start:.2f} sec")

    cfModel = RecommenderSystem(random_seed=71070)

    print("load model parameters")
    start = time.time()
    if model_name == "svd++":
        svd_params = load_advanced_svd_model(model_path)
        cfModel.advanced_model = SVDModelEngine(svd_params)
    elif model_name == "svd":
        svd_params = load_svd_model(model_path)
        cfModel.base_model = SVDModelEngine(svd_params)
    else:
        raise Exception(f"Not a valid model name: {model_name}")

    end = time.time()
    print(f"loading model parameters took {end - start:.2f}")

    # evaluate on train data
    y_true = train_df.stars * 5
    y_pred = cfModel.PredictRating(train_df, model_name)
    train_score = rmse(y_true, y_pred)
    print(f"train - rmse: {train_score}")

    print("Evaluate Testset")
    y_true = test_df.stars * 5

    # evaluate random
    y_random = np.asarray([random.choice([0.2, 0.4, 0.6, 0.8, 1.0]) for _ in range(len(y_true))]) * 5
    random_score = rmse(y_true, y_random)
    print(f"random - rmse: {random_score}")

    # evaluate constant
    mean = np.mean(y_true)
    y_mean = [mean for _ in range(len(y_true))]
    mean_score = rmse(y_true, y_mean)
    print(f"mean - rmse: {mean_score}")

    #evaluate baseline


    # evaluate on test data
    y_pred = cfModel.PredictRating(test_df, model_name) * 5
    test_score = rmse(y_true, y_pred)
    print(f"test - rmse: {test_score}")

