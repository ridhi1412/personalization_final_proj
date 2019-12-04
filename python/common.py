# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:05:32 2019

@author: rmahajan14
"""
import os
import pandas as pd

from pyspark import SparkContext
from pyspark.sql import SQLContext

DIRPATH = r'P:\rmahajan14\columbia\fall 2019\Personalization\final_project'
EXCEL_PATH = os.path.join(DIRPATH, 'data')
CACHE_PATH = os.path.join(DIRPATH, 'cache')

import json
from tqdm.auto import tqdm

def pandas_to_spark(pandas_df):
    sc = SparkContext.getOrCreate()  # else get multiple contexts error
    sql_sc = SQLContext(sc)
    spark_df = sql_sc.createDataFrame(pandas_df)
    return spark_df


def load_pandas(file_name='review.json'):
    line_count = len(
        open(os.path.join(EXCEL_PATH, file_name), encoding='utf8').readlines())
    user_ids, business_ids, stars, dates = [], [], [], []
    with open(os.path.join(EXCEL_PATH, file_name), encoding='utf8') as f:
        for line in tqdm(f, total=line_count):
            blob = json.loads(line)
            user_ids += [blob["user_id"]]
            business_ids += [blob["business_id"]]
            stars += [blob["stars"]]
            dates += [blob["date"]]

    ratings = pd.DataFrame({
        "user_id": user_ids,
        "business_id": business_ids,
        "rating": stars,
        "date": dates
    })
    user_counts = ratings["user_id"].value_counts()
    active_users = user_counts.loc[user_counts >= 5].index.tolist()
    return ratings, user_counts, active_users


if __name__ == '__main__':
    ratings, user_counts, active_users = load_pandas()
