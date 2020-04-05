import sys
import os
module_dir = os.path.dirname(os.path.dirname(__file__))
print(module_dir)
sys.path.append(module_dir)

from recsys.utils.data.yelp_dataset import load_yelp_dataset, split_dataset, reindex_data
from recsys.cf import RecommenderSystem
from recsys.cf.basemodel import BaseSVDModelParams, save_base_svd_model_parameters

import time


if __name__ == "__main__":
    # sys.argv[1]
    train_path = r"C:\\Users\\ronp\\Documents\\מסמכים לתואר שני\\recsys\\ex1\\data\\trainData.csv"
    # sys.argv[2]
    test_path = r"C:\\Users\\ronp\\Documents\\מסמכים לתואר שני\\recsys\\ex1\\data\\testData.csv"
    num_latent_features = 40  # int(sys.argv[3])
    advanced_model = False # bool(sys.argv[4])
    save_model_file = "base-svd-model"
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

    CFModel = RecommenderSystem(random_seed=71070)
    if advanced_model:
        print("train advanced model")
        CFModel.TrainAdvancedModel(train_df, num_latent_features)
    else:
        print("train base model")
        CFModel.TrainBaseModel(train_df, num_latent_features)
        
        if save_model_file is not None:
            save_base_svd_model_parameters(CFModel.base_model.model_parameters_, save_model_file)
            

