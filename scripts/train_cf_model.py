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


# "/home/ron/data/studies/bgu/recsys/ex1/data/trainData.csv" \
    # "/home/ron/data/studies/bgu/recsys/ex1/data/testData.csv" \
    # "--model-name svd" \
    # "--model-path /home/ron/data/studies/bgu/recsys/ex1/data/svd++100-model.npz"
    # sys.argv[1]
    # train_path =
    # sys.argv[2]
    # test_path = ""

    # model_name = "svd"  # sys.argv[3]

    # model_path = "/home/ron/data/studies/bgu/recsys/ex1/data/base-svd-model.npz"  # sys.argv[4]
    # model_path = "/home/ron/data/studies/bgu/recsys/ex1/data/advanced-svd-model.npz" # sys.argv[4]
    # model_path = "/home/ron/data/studies/bgu/recsys/ex1/data/svd++100-model.npz" # sys.argv[4]
