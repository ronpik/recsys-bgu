import random
import sys
import os
import time
from typing import Sequence, Any

import numpy as np
import pandas as pd


module_dir = os.path.dirname(os.path.dirname(__file__))
print(module_dir)
sys.path.append(module_dir)

from recsys.utils.data.yelp_dataset import load_yelp_dataset, reindex_data
from recsys.cf import RecommenderSystem, SVDModelEngine
from recsys.cf.basemodel import load_svd_model, save_base_svd_model
from recsys.cf.advancedmodel import load_advanced_svd_model, save_advanced_svd_model
from recsys.eval.evaltools import rmse, average_ndpm

import argparse


RANDOM_SEED = 71070
RUN_TRAIN = "train"
RUN_EVAL = "eval"
SVD_MODEL = "svd"
SVD_ADVANCED_MODEL = "svd++"


def initialize(train_path: str, test_path: str):
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
    return train_df, test_df


def run_training(model_name: str, train_df: pd.DataFrame, num_latent_features: int) -> RecommenderSystem:
    cf_model = RecommenderSystem(random_seed=71070)
    print(f"train model {model_name}")
    if model_name == SVD_MODEL:
        print("train base model")
        cf_model.TrainBaseModel(train_df, num_latent_features)
        cf_model.PredictRating(test_df, SVD_MODEL)

    elif model_name == SVD_ADVANCED_MODEL:
        cf_model.TrainAdvancedModel(train_df, num_latent_features)
        cf_model.PredictRating(test_df, SVD_ADVANCED_MODEL)

    return cf_model


def run_evaluation(train_df: pd.DataFrame, test_df: pd.DataFrame, cf_model: RecommenderSystem, model_name: str):
    print(f"\nRun evaluation on test-set with the model {model_name}")
    y_true = test_df.stars * 5
    users = test_df.user_id

    print(f"Evaluate model: {model_name}")
    print("performing predictions on train data")
    y_test_pred = np.multiply(5, cf_model.PredictRating(test_df, model_name))
    evaluate(y_true, y_test_pred, users)
    print("\nEvaluate random")
    y_random = np.asarray([random.choice([0.2, 0.4, 0.6, 0.8, 1.0]) for _ in range(len(y_true))]) * 5
    evaluate(y_true, y_random, users)
    print("\nEvaluate with constant (mean rating) ")
    mean = np.mean(y_true)
    y_mean = np.repeat(mean, len(y_true))
    evaluate(y_true, y_mean, users)
    print(f"\nEvaluate on train-set")
    y_train_pred = np.multiply(5, cf_model.PredictRating(train_df, model_name))
    evaluate(y_true, y_train_pred, users)


def evaluate(y_true: Sequence[int], y_pred: Sequence[float], users: Sequence[Any]):
    print("calculating RMSE on predictions")
    train_score = rmse(y_true, y_pred)
    print(f"rmse: {train_score}")
    print("calculating NDPM on predictions")
    ndpm_score = average_ndpm(y_true, y_pred, users)
    print(f"ndpm: {ndpm_score}")


def save_model(model_name: str, cf_model: RecommenderSystem, outpath: str):
    print(f"save trained model {model_name} to {os.path.abspath(outpath)}")
    if model_name == SVD_MODEL:
        save_base_svd_model(cf_model.advanced_model.model_parameters_, outpath)
    elif model_name == SVD_ADVANCED_MODEL:
        save_advanced_svd_model(cf_model.advanced_model.model_parameters_, outpath)


def load_model(model_name: str, model_path: str) -> RecommenderSystem:
    cf_model = RecommenderSystem(random_seed=RANDOM_SEED)
    print("load model parameters")
    start = time.time()
    if model_name == "svd++":
        svd_params = load_advanced_svd_model(model_path)
        cf_model.advanced_model = SVDModelEngine(svd_params)
    elif model_name == "svd":
        svd_params = load_svd_model(model_path)
        cf_model.base_model = SVDModelEngine(svd_params)
    else:
        raise Exception(f"Not a valid model name: {model_name}")
    end = time.time()
    print(f"loading model parameters took {end - start:.2f}")

    return cf_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(action=RUN_TRAIN)
    train_parser.add_argument("trainset", type=str)
    train_parser.add_argument("testset", type=str)
    train_parser.add_argument("--model-name", "-m", type=str, default="svd",
                              help="name of the model to use to perform factorization")
    train_parser.add_argument("--num-latent", "-l", type=int, default=50,
                              help="number of latent features in the factorization for users and items")
    train_parser.add_argument("--outpath", "-o", type=int, default="cf_model",
                              help="a path where to store the trained model")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.set_defaults(action=RUN_EVAL)
    eval_parser.add_argument("trainset", type=str)
    eval_parser.add_argument("testset", type=str)
    eval_parser.add_argument("--model-name", "-m", type=str, default="svd",
                             help="name of the model to be loaded")
    eval_parser.add_argument("--model-path", "-p", type=str,
                             help="path from where to load the model")

    args = parser.parse_args()

    train_df, test_df = initialize(args.trainset, args.testset)

    cf_model = None
    if args.action == RUN_TRAIN:
        print("Run in training mode (including evaluation)")
        cf_model = run_training(args.model_name, train_df, args.num_latent)
        if args.outpath is not None:
            save_model(args.model_name, cf_model, args.outpath)

    elif args.action == RUN_EVAL:
        print("run in evaluation only mode")
        cf_model = load_model(args.model_name, args.model_path)

    run_evaluation(train_df, test_df, cf_model, args.model_name)
