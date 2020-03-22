import sys

from recsys.utils.data import get_yelp_data_for_cf

if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    train_mat, test_mat = get_yelp_data_for_cf(train_path, test_path)

    print(f"train data of type: {type(train_mat)} with shape: {train_mat.shape}")
    print(f"test data of type: {type(test_mat)} with shape: {test_mat.shape}")

