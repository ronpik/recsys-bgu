import sys
import time

from recsys.cf import RecommenderSystem
from recsys.utils.data import get_yelp_data_for_cf

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    p = int(sys.argv[3])
    q = int(sys.argv[4])
    start = time.time()
    train_mat, test_mat = get_yelp_data_for_cf(train_path, test_path)
    end = time.time()
    print(f"loading data took {end - start}s")

    print(f"train data of type: {type(train_mat)} with shape: {train_mat.shape}")
    print(f"test data of type: {type(test_mat)} with shape: {test_mat.shape}")

    CFModel = RecommenderSystem()
    CFModel.TrainBaseModel(train_mat, p, q)

