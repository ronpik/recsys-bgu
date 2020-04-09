import sys
import os

from recsys.cf.advancedmodel import save_advanced_svd_model

module_dir = os.path.dirname(os.path.dirname(__file__))
print(module_dir)
sys.path.append(module_dir)

from recsys.utils.data.yelp_dataset import load_yelp_dataset, reindex_data
from recsys.cf import RecommenderSystem
from recsys.cf.basemodel import save_base_svd_model

import time

if __name__ == "__main__":
    # sys.argv[1]
    train_path = "/home/ron/data/studies/bgu/recsys/ex1/data/trainData.csv"
    # sys.argv[2]
    test_path = "/home/ron/data/studies/bgu/recsys/ex1/data/testData.csv"
    num_latent_features = 100  # int(sys.argv[3])
    advanced_model = True  # bool(sys.argv[4])
    save_model_file = "/home/ron/data/studies/bgu/recsys/ex1/data/svd++100-model.npz"
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
    if advanced_model:
        print("train advanced model")
        cfModel.TrainAdvancedModel(train_df, num_latent_features)
        cfModel.PredictRating(test_df, "svd++")

        if save_model_file is not None:
            print(f"save base model parameters to {os.path.abspath(save_model_file)}")
            save_advanced_svd_model(cfModel.advanced_model.model_parameters_, save_model_file)

    else:
        print("train base model")
        cfModel.TrainBaseModel(train_df, num_latent_features)
        cfModel.PredictRating(test_df, "svd")

        if save_model_file is not None:
            print(f"save base model parameters to {os.path.abspath(save_model_file)}")
            save_base_svd_model(cfModel.base_model.model_parameters_, save_model_file)
