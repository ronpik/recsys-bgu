import csv
from typing import Dict, List, Tuple
import random

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, coo_matrix

COLS = list(range(5))
COLS_NO_INDEX = COLS[1:]
COLS_NO_TEXT = COLS[:-1]


USER_ID_FIELD = "user_id"
BUSINESS_ID_FIELD = "business_id"
RATING_FIELD = "stars"

# sampling users for validation
SPLIT_AVAILABLE = 2


def load_yelp_dataset(path: str, use_text=False) -> pd.DataFrame:
    if use_text:
        df = pd.read_csv(path, header=0, index_col=0)
    else:
        df = pd.read_csv(path, header=0, usecols=COLS_NO_TEXT, index_col=0)
    print(f"loaded data with size: {df.shape}")
    df.dropna(axis=0, inplace=True)
    print(f"filtered NaN values")
    df[RATING_FIELD] = df[RATING_FIELD].astype(np.uint8)
    return df


def prepare_data_for_cf(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[spmatrix, spmatrix]:
    train_size = train_df.shape[0]
    merged_df = train_df.append(test_df)
    del train_df
    del test_df

    print("index users")
    index_users_col = "user_index"
    index_by_unique_elements(merged_df, USER_ID_FIELD)

    print("index business ids")
    business_users_col = "business_index"
    index_by_unique_elements(merged_df, BUSINESS_ID_FIELD)

    print("resplit to train and test")
    train_df = merged_df[:train_size]
    test_df = merged_df[train_size:]
    del merged_df

    print("convert TRAIN to sparse matrix")
    train_indices_df = train_df[[
        index_users_col, business_users_col, RATING_FIELD]]
    del train_df
    train_mat = df_to_sparse(train_indices_df)
    del train_indices_df

    print("convert TEST to sparse matrix")
    test_indices_df = test_df[[index_users_col,
        business_users_col, RATING_FIELD]]
    del test_df
    train_rows, train_cols = train_mat.shape
    test_mat = df_to_sparse(test_indices_df, train_rows - 1, train_cols - 1)
    del test_indices_df

    return train_mat, test_mat


def reindex_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_size = train_df.shape[0]
    merged_df = train_df.append(test_df)
    del train_df
    del test_df

    print("index users")
    index_by_unique_elements(merged_df, USER_ID_FIELD)
    print("index business ids")
    index_by_unique_elements(merged_df, BUSINESS_ID_FIELD)
    print("resplit to train and test")
    train_df = merged_df[:train_size]
    test_df = merged_df[train_size:]
    del merged_df

    print("remove records from test that doesn't occur in train")
    max_user_index = np.max(train_df[USER_ID_FIELD])
    max_item_index = np.max(train_df[BUSINESS_ID_FIELD])

    test_mask_by_users = test_df[USER_ID_FIELD] <= max_user_index
    test_mask_by_items = test_df[BUSINESS_ID_FIELD] <= max_item_index
    test_mask = np.logical_and(test_mask_by_users, test_mask_by_items)
    test_df = test_df[test_mask]
    return train_df, test_df


def index_by_unique_elements(data: pd.DataFrame, column_name: str):
    """
    assign index to each element and integrate into the data-frame.
    :return:
    """
    indices: Dict[str, int] = {}
    indexed_col: List[int] = [indices.setdefault(
        e, len(indices)) for e in data[column_name]]
    data[column_name] = np.array(indexed_col, dtype=np.uint32)


def df_to_sparse(df: pd.DataFrame, max_row_index: int = None, max_col_index: int = None) -> spmatrix:
    """
    convert data frame into sparse matrix. by convention df should contains only 3 columns -
    the first for the row index, the second for the col index and the last for the value.
    :return:
    """
    mask = np.full(df.shape[0], fill_value=True)
    if max_row_index is not None:
        rows_in_range_mask = df.iloc[:, 0] <= max_row_index
        mask = np.logical_and(mask, rows_in_range_mask)
    else:
        max_row_index = df.iloc[:, 0].max()

    if max_col_index is not None:
        cols_in_range_mask = df.iloc[:, 1] <= max_col_index
        mask = np.logical_and(mask, cols_in_range_mask)
    else:
        max_col_index = df.iloc[:, 1].max()

    df = df[mask]

    print("\tdf to coo matrix")
    rows = df.iloc[:, 0]
    cols = df.iloc[:, 1]
    data = df.iloc[:, 2]
    coo_mat = coo_matrix((data, (rows, cols)), shape=(
        max_row_index + 1, max_col_index + 1))
    print("\tcoo to csr matrix")
    return coo_mat.tocsr()


def split_dataset(df: pd.DataFrame,
    users_size: float,
    items_per_user: float,
    random_seed: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    split a given dataset into two different datasets, having different elements.
    returns two datases, the first has the rows of the given dataset except those that where splitted accoding to the other parameters.
    users size the the ratio of users to sample items from, and the items_per_user is the ratio of elements to split from the main dataset
    for each of the chosen users.
    """
    if items_per_user >= 1:
        raise ValueError(
            f"users_size should be a number greater than 0 and smaller than 1: {items_per_user}")

    random_generator = random.Random(random_seed)

    unique_users = df[USER_ID_FIELD].unique()
    unique_items = df[BUSINESS_ID_FIELD].unique()
    
    # ensure that users_size is a ratio (not a absolute size)
    if users_size > 1:
        users_size = float(users_size) / len(unique_users)

    # assuming users are actually the index of the users.
    users_sample = np.random.choice(
        a = [0, 1], \
        size = len(unique_users), \
        p = [users_size, 1-users_size]
    )

    split_available_items = np.zeros(len(unique_items))

    # validate that all users and items in the splitted dataset are still exist in the train dataset
    def choose_item_rating(user: int, item: int) -> bool:
        splitable_item = True
        splitable_user = True

        if not split_available_items[item]:
            split_available_items[item] = SPLIT_AVAILABLE
            splitable_item = False
        
        if not users_sample[user]:
            return False

        if users_sample[user] != SPLIT_AVAILABLE:
            users_sample[user] = SPLIT_AVAILABLE
            splitable_user = False

        if (splitable_item and splitable_user):
            return random_generator.random() < items_per_user
        
        return False

    split_mask=np.full(len(df.index), fill_value = False)
    for i, (user_id, item_id, _) in enumerate(df.itertuples(index=False, name=None)):
        if choose_item_rating(user_id, item_id):
            split_mask[i] = True

    reduced_df = df[~split_mask]
    split_df = df[split_mask]
    return reduced_df, split_df


def get_yelp_data_for_cf(train_path: str, test_path: str) -> Tuple[spmatrix, spmatrix]:
    print(f"load train data: {train_path}")
    train_df=load_yelp_dataset(train_path)
    print(f"load test data: {train_path}")
    test_df=load_yelp_dataset(test_path)
    train_mat, test_mat=prepare_data_for_cf(train_df, test_df)
    return train_mat, test_mat

