# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:46:40 2019

@author: rmahajan14
"""

import os
import pandas as pd

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode
from time import time

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from common import CACHE_PATH, EXCEL_PATH
    from common import load_pandas, pandas_to_spark
except:
    from python.common import CACHE_PATH, EXCEL_PATH
    from python.common import load_pandas, pandas_to_spark

#try:
#    from sample_df import sample_df_threshold_use_pandas
#except:
#    from utils.sample_df import sample_df_threshold_use_pandas


def get_als_model(df,
                  rank,
                  regParam=1,
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
                        regParam=regParam,
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


def get_best_rank(df,
                  ranks=[2**i for i in range(7)],
                  regParams=[0.005, 0.05, 0.5, 5]):
    """
        Returns a report of performance metrics for ALS model for diffrent ranks
    """
    rmse_train_dict = dict()
    coverage_train_dict = dict()
    rmse_test_dict = dict()
    coverage_test_dict = dict()
    running_time_dict = dict()

    # for rank in ranks:
    #     _, model, rmse_train, rmse_test, coverage_train, coverage_test, running_time, _, _ = get_als_model(
    #         df, rank, model='ALS', evaluator='Regression')
    #     rmse_train_dict[rank] = rmse_train
    #     rmse_test_dict[rank] = rmse_test
    #     coverage_train_dict[rank] = coverage_train
    #     coverage_test_dict[rank] = coverage_test
    #     running_time_dict[rank] = running_time

    #TODO Remove this
    rank = 64
    for regParam in regParams:
        (_, model, rmse_train, rmse_test, coverage_train, coverage_test,
         running_time, _, _) = get_als_model(df,
                                             rank,
                                             regParam,
                                             model='ALS',
                                             evaluator='Regression')
        rmse_train_dict[regParam] = rmse_train
        rmse_test_dict[regParam] = rmse_test
        coverage_train_dict[regParam] = coverage_train
        coverage_test_dict[regParam] = coverage_test
        running_time_dict[regParam] = running_time

    df = pd.DataFrame(data=np.asarray([
        list(rmse_train_dict.keys()),
        list(rmse_train_dict.values()),
        list(rmse_test_dict.values()),
        list(coverage_train_dict.values()),
        list(coverage_test_dict.values()),
        list(running_time_dict.values())
    ]).T,
                      columns=[
                          'Rank', 'RMSE_train', 'RMSE_test', 'Coverage_train',
                          'Coverage_test', 'Running_time'
                      ])

    return df


def plot_performance_als(report_df, report_type='rank'):
    if report_type == 'rank':
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))

        ax[0].plot(report_df['RMSE_train'])
        ax[0].plot(report_df['RMSE_test'])
        ax[0].legend(['Train', 'Test'])
        ax[0].title.set_text('Error vs Rank for ALS model')
        ax[0].set_ylabel('RMSE')
        ax[0].set_xlabel('Log_2(Rank)')

        ax[1].plot(report_df['Coverage_train'])
        ax[1].plot(report_df['Coverage_test'])
        ax[1].legend(['Train', 'Test'])
        ax[1].title.set_text('Coverage vs Rank for ALS model')
        ax[1].set_ylabel('Coverage')
        ax[1].set_xlabel('Log_2(Rank)')
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.plot(report_df['Running_time'])
        plt.title('Running Time vs Rank for ALS model')
        plt.ylabel('Running Time (seconds)')
        plt.xlabel('Log_2(Rank)')
        plt.show()

        print('\n')

    elif report_type == 'sample':
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        low = int(math.log10(report_df['Sample_size'].iloc[0]))
        high = int(math.log10(report_df['Sample_size'].iloc[-1]))
        x_tick = [i for i in range(low, high)]
        ax[0].plot(report_df['RMSE_train'])
        ax[0].plot(report_df['RMSE_test'])
        ax[0].legend(['Train', 'Test'])
        ax[0].title.set_text('Error vs Sample Size for ALS model')
        ax[0].set_ylabel('RMSE')
        ax[0].set_xlabel('Log_10(Sample_size)+4')
        #ax[0].set_xticks(x_tick)

        ax[1].plot(report_df['Coverage_train'])
        ax[1].plot(report_df['Coverage_test'])
        ax[1].legend(['Train', 'Test'])
        ax[1].title.set_text('Coverage vs Sample Size for ALS model')
        ax[1].set_ylabel('Coverage')
        ax[1].set_xlabel('Log_10(Sample_size)+4')
        #ax[1].set_xticks(x_tick)
        plt.show()

        plt.figure(figsize=(20, 5))
        plt.plot(report_df['Running_time'])
        plt.title('Running Time vs Sample Size for ALS model')
        plt.ylabel('Running Time (seconds)')
        plt.xlabel('Log_10(Sample_size)+4')
        #plt.xticks(x_tick)
        plt.show()

        print('\n')


if __name__ == '__main__':
    frac = 0.001
    df, _, _ = load_pandas()
    print('Getting df')
    df = df.sample(frac=frac, random_state=0)
    print('Got df')
    start = time()
    # (predictions, model, rmse_train, rmse_test, coverage_train, coverage_test,
    #  running_time) = get_als_model(df, 5)
    pow_two_max_rank = 6
    ranks = [2**i for i in range(pow_two_max_rank + 1)]
    report_df = get_best_rank(df, ranks=ranks)
    running_time = time() - start
    print(f'running time = {running_time}')
