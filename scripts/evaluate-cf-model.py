import sys
import os
module_dir = os.path.dirname(os.path.dirname(__file__))
print(module_dir)
sys.path.append(module_dir)

import time

from recsys.utils.data.yelp_dataset import load_yelp_dataset, split_dataset, reindex_data
from recsys.cf import RecommenderSystem, SVDModelEngine
from recsys.cf.basemodel import BaseSVDModelParams, save_base_svd_model_parameters, load_svd_model
from recsys.eval.evaltools import rmse


if __name__ == "__main__":
    # sys.argv[1]
    train_path = r"C:\\Users\\ronp\\Documents\\מסמכים לתואר שני\\recsys\\ex1\\data\\trainData.csv"
    # sys.argv[2]
    test_path = r"C:\\Users\\ronp\\Documents\\מסמכים לתואר שני\\recsys\\ex1\\data\\testData.csv"
    advanced_model = False # bool(sys.argv[3])
    model_path = r"C:\\Users\\ronp\\PycharmProjects\\recsys\\recsys\\cf\\base-svd-model.npz" # sys.argv[4]
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
    if advanced_model:
        svd_params = None #load_svd_model(model_path)
        cfModel.advanced_model = SVDModelEngine(svd_params)
    else:
        svd_params = load_svd_model(model_path)
        cfModel.base_model = SVDModelEngine(svd_params)
    end = time.time()
    print(f"loading model parameters took {end - start:.2f}")

    y_true = test_df.stars
    y_pred = cfModel.PredictRating(test_df, "svd")
    score = rmse(y_true, y_pred)
    print(f"rmse: {score}")

