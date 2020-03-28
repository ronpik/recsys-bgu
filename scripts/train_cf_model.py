import sys
import time

# sys.path.append('../recsys')
sys.path.append("c:/Users/ronp/PycharmProjects/recsys")
from recsys.cf import RecommenderSystem
from recsys.utils.data.yelp_dataset import load_yelp_dataset, split_dataset

if __name__ == "__main__":
    train_path = r"C:\\Users\\ronp\Documents\\מסמכים לתואר שני\\recsys\\ex1\data\\trainData.csv" #sys.argv[1] 
    test_path = r"C:\\Users\\ronp\Documents\\מסמכים לתואר שני\\recsys\\ex1\data\\testData.csv" #sys.argv[2]
    num_latent_features = 40 #int(sys.argv[3])
    start = time.time()
    print(f"load train data: {train_path}")
    train_df = load_yelp_dataset(train_path)
    print(f"load test data: {train_path}")
    test_df = load_yelp_dataset(test_path)
    end = time.time()
    print(f"loading data took {end - start:.2f}s")

    print("splitting train to validation")
    start = time.time()
    train_df, validation_df = split_dataset(train_df, 0.5, 0.2)
    end = time.time()
    print(f"splitting to validation took {end - start:.2f}sec")

    print(f"train data loaded with shape: {train_df.shape}")
    print(f"test data loaded with shape: {test_df.shape}")

    CFModel = RecommenderSystem(train_df, test_df)
    CFModel.TrainBaseModel(num_latent_features)

