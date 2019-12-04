import pandas as pd
import numpy as np
import surprise
from surprise import Reader
from surprise.model_selection import cross_validate
#import os
from common import CACHE_PATH, EXCEL_PATH, load_pandas


def baseline_bias_model(df):
    """
        Shows the performance of model based on just bias
    """
    ratings_pandas_df = df.drop(columns=['date'])
#    ratings_pandas_df.columns = ['user_id', 'business_id', 'rating']

    reader = Reader(rating_scale=(1, 5))  #TODO figure out
    data = surprise.dataset.Dataset.load_from_df(
        df=ratings_pandas_df, reader=reader)
    _ = cross_validate(
        surprise.prediction_algorithms.baseline_only.BaselineOnly(),
        data,
        measures=['RMSE', 'MAE'],
        cv=5,
        verbose=1,
        n_jobs=-1)

    print('\n')


if __name__ == '__main__':
    frac = 1
    df, _, _ = load_pandas()
    df = df.sample(frac=frac, random_state=0)
    baseline_bias_model(df)
