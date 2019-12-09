# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:46:40 2019

@author: rmahajan14
"""

from common import CACHE_PATH, EXCEL_PATH
import os
import pandas as pd

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode
from time import time
from common import CACHE_PATH, EXCEL_PATH
from common import load_pandas, pandas_to_spark

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#try:
#    from sample_df import sample_df_threshold_use_pandas
#except:
#    from utils.sample_df import sample_df_threshold_use_pandas


def get_als_model(df,
                  rank,
                  split=[0.8, 0.2],
                  model='ALS',
                  evaluator='Regression',
                  use_cache=True):

    cache_path = os.path.join(CACHE_PATH, f'get_als_model.msgpack')
    if use_cache and os.path.exists(cache_path):
        print(f'Loading from {cache_path}')
        (predictions, model, rmse_train, rmse_test, coverage_train,
         coverage_test, running_time, train,
         test) = pd.read_msgpack(cache_path)
        print(f'Loaded from {cache_path}')
    else:

        le1 = LabelEncoder()
        le1.fit(df['user_id'])
        df['user_id'] = le1.transform(df['user_id'])
        print(len(df['user_id']))
        le2 = LabelEncoder()
        le2.fit(df['business_id'])
        df['business_id'] = le2.transform(df['business_id'])
        print(len(df['business_id']))

        df = pandas_to_spark(df)

        train, test = df.randomSplit(split, seed=1)

        total_unique_businessids_train = train.select(
            'business_id').distinct().toPandas().values
        total_unique_businessids_test = test.select(
            'business_id').distinct().toPandas().values

        if model == 'ALS':
            model = ALS(maxIter=5,
                        regParam=0.09,
                        rank=rank,
                        userCol="user_id",
                        itemCol="business_id",
                        ratingCol="rating",
                        coldStartStrategy="drop",
                        nonnegative=True)

        if evaluator == 'Regression':
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
        start = time()
        model = model.fit(train)
        running_time = time() - start
        predictions = model.transform(test)
        rmse_test = evaluator.evaluate(model.transform(test))
        rmse_train = evaluator.evaluate(model.transform(train))

        pred_unique_businessids = calculate_coverage(model)
        subset_pred_train = [
            i for i in pred_unique_businessids
            if i in total_unique_businessids_train
        ]
        subset_pred_test = [
            i for i in pred_unique_businessids
            if i in total_unique_businessids_test
        ]
        coverage_train = len(subset_pred_train) / len(
            total_unique_businessids_train)
        coverage_test = len(subset_pred_test) / len(
            total_unique_businessids_test)

        #        pd.to_msgpack(cache_path, (predictions, model, rmse_train, rmse_test, coverage_train,
        #            coverage_test, running_time, train, test))
        print(f'Dumping to {cache_path}')

    # breakpoint()

    return (predictions, model, rmse_train, rmse_test, coverage_train,
            coverage_test, running_time, train, test)


def calculate_coverage(model):
    """
        Returns all unique movies ids recommended atleast once to a user
    """
    user_recos = model.recommendForAllUsers(numItems=10)
    df1 = user_recos.select(explode(user_recos.recommendations).alias('col1'))
    df2 = df1.select('col1.*')
    df3 = df2.select('business_id').distinct()
    df4 = df3.toPandas()
    movie_set = df4['business_id'].values
    return movie_set


if __name__ == '__main__':
    frac = 0.001
    df, _, _ = load_pandas()
    print('Getting df')
    df = df.sample(frac=frac, random_state=0)
    print('Got df')
    start = time()
    (predictions, model, rmse_train, rmse_test, coverage_train, coverage_test,
     running_time) = get_als_model(df, 5)
    running_time = time() - start
    print(f'running time = {running_time}')
