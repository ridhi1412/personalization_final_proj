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
    lower = ratings['date_dist_rating'].min()
    upper = ratings['date_dist_rating'].max()
    df = df

    reader = surprise.Reader(rating_scale=(lower1, upper1))  
    data = surprise.dataset.Dataset.load_from_df(
        df=df, reader=reader)

    alg = surprise.SVDpp()
    results = cross_validate(alg, data, measures=['RMSE'], cv=3)
    print(results)


if __name__ == '__main__':
    frac = 1
    df = pd.read_csv("../data/time_location_aware.csv")
    df = df.sample(frac=frac, random_state=0)
    baseline_bias_model(df)