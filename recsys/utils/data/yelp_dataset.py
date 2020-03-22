import csv
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, coo_matrix

COLS = list(range(5))
COLS_NO_INDEX = COLS[1:]
COLS_NO_TEXT = COLS[:-1]


USER_ID_FIELD = "user_id"
BUSINESS_ID_FIELD = "business_id"
RATING_FIELD = "stars"


def load_yelp_dataset(path: str) -> pd.DataFrame:
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
    index_by_unique_elements(merged_df, USER_ID_FIELD, index_users_col)

    print("index business ids")
    business_users_col = "business_index"
    index_by_unique_elements(merged_df, BUSINESS_ID_FIELD, business_users_col)

    print("resplit to train and test")
    train_df = merged_df[:train_size]
    test_df = merged_df[train_size:]
    del merged_df

    print("convert TRAIN to sparse matrix")
    train_indices_df = train_df[[index_users_col, business_users_col, RATING_FIELD]]
    del train_df
    train_mat = df_to_sparse(train_indices_df)
    del train_indices_df

    print("convert TEST to sparse matrix")
    test_indices_df = test_df[[index_users_col, business_users_col, RATING_FIELD]]
    del test_df
    test_mat = df_to_sparse(test_indices_df)
    del test_indices_df

    return train_mat, test_mat


def index_by_unique_elements(data: pd.DataFrame, column_name: str, new_col_name: str):
    """
    assign index to each element and integrate into the data-frame.
    :return:
    """
    indices: Dict[str, int] = {}
    indexed_col: List[int] = [indices.setdefault(e, len(indices)) for e in data[column_name]]
    data[new_col_name] = indexed_col


def df_to_sparse(df: pd.DataFrame) -> spmatrix:
    """
    convert data frame into sparse matrix. by convention df should contains only 3 columns -
    the first for the row index, the second for the col index and the last for the value.
    :param df:
    :return:
    """
    print("\tdf to coo matrix")
    coo_mat = coo_matrix(df, dtype=np.uint8)
    print("\tcoo to csr matrix")
    return coo_mat.tocsr()


def get_yelp_data_for_cf(train_path: str, test_path: str) -> Tuple[spmatrix, spmatrix]:
    print(f"load train data: {train_path}")
    train_df = load_yelp_dataset(train_path)
    print(f"load test data: {train_path}")
    test_df = load_yelp_dataset(test_path)
    train_mat, test_mat = prepare_data_for_cf(train_df, test_df)
    return train_mat, test_mat



