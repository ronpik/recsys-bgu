import sys
import os
module_dir = os.path.abspath("./../")
sys.path.append(module_dir)
import time

from recsys.cf import RecommenderSystem
from recsys.utils.data.yelp_dataset import load_yelp_dataset

#if __name__ == "__main__":
#    train_path = sys.argv[1]
#    test_path = sys.argv[2]


train_path = '/Users/shaimeital/code/thesis/recsys-bgu/data/trainData.csv'
test_path = '/Users/shaimeital/code/thesis/recsys-bgu/data/testData.csv'

num_latent_features = 40 # int(sys.argv[3])
start = time.time()
print(f"load train data: {train_path}")
train_df = load_yelp_dataset(train_path)
print(f"load test data: {train_path}")
test_df = load_yelp_dataset(test_path)
end = time.time()
print(f"loading data took {end - start}s")

print(f"train data loaded with shape: {train_df.shape}")
print(f"test data loaded with shape: {test_df.shape}")

CFModel = RecommenderSystem(train_df, test_df)
CFModel.TrainAdvancedModel(num_latent_features)

#tests




