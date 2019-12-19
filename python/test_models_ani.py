# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 03:20:02 2019

@author: aksmi
"""

import getpass
user = getpass.getuser()

if user == 'aksmi':
    import findspark
    findspark.init()

from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode
from pyspark.sql import SQLContext

from time import time

import scipy.sparse as sp
import numpy as np
import pandas as pd

import gc
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import numpy as np
from sklearn.model_selection import train_test_split

try:
    from common import load_pandas, pandas_to_spark
    from common import CACHE_PATH, EXCEL_PATH
    from colab_filtering_basline2 import get_als_model
    from bias1 import baseline_bias_model, get_tr_te_pr
    from content_based import content_based
    from Deep_learning import DL_Model
    from time_location import time_location_model, get_tr_te_pr_time_loc

    from hermes import (calculate_serendipity, calculate_novelty,
                        calculate_novelty_bias, calculate_prediction_coverage,
                        calculate_rmse_using_rdd)
except:
    from python.common import load_pandas, pandas_to_spark
    from python.common import CACHE_PATH, EXCEL_PATH
    from python.colab_filtering_basline2 import get_als_model
    from python.bias1 import baseline_bias_model, get_tr_te_pr
    from python.content_based import content_based
    from python.Deep_learning import DL_Model
    from python.time_location import time_location_model, get_tr_te_pr_time_loc

    from python.hermes import (calculate_serendipity, calculate_novelty,
                               calculate_novelty_bias,
                               calculate_prediction_coverage,
                               calculate_rmse_using_rdd)
    
# count_unique_biz_total = 0


def get_create_context():
    sc = SparkContext.getOrCreate()  # else get multiple contexts error
    sqlCtx = SQLContext(sc)

    return sc, sqlCtx


def create_test_train(train, test):
    x_cols = ['user_id', 'business_id', 'rating']
    y_cols = ['rating']

    X_train = train.select(x_cols)
    y_train = train.select(y_cols)

    X_test = test.select(x_cols)
    y_test = test.select(y_cols)

    return (X_train, X_test, y_train, y_test)


def get_small_sample_df(frac=0.001, prolific=10):
    df, _, _ = load_pandas()
    print('Getting df')
    # df.user_id.value_count < prolific
    df = df.sample(frac=frac, random_state=0)
    global count_unique_biz_total
    count_unique_biz_total = len(list(df['business_id'].unique()))
    # breakpoint()
    print('Got df')
    return df


def get_als():

    df = get_small_sample_df()

    # Get predictions from ALS
    (y_predicted, model, rmse_train, rmse_test, coverage_train, coverage_test,
     running_time, train, test) = get_als_model(df, 5)

    # Get train test
    X_train, X_test, y_train, y_test = create_test_train(train, test)

    predictions = y_predicted.select(['user_id', 'business_id', 'prediction'])

    return (X_train, X_test, y_train, y_test, predictions)


def get_bias():
    df = get_small_sample_df()

    (trainset, testset, predictions, dusers, ditems) = baseline_bias_model(df)
    df_train, df_test, df_pred = get_tr_te_pr(trainset, testset, predictions,
                                              dusers, ditems)

    X_train = pandas_to_spark(df_train)
    X_test = pandas_to_spark(df_test)
    predictions = pandas_to_spark(df_pred)

    X_train, X_test, y_train, y_test = create_test_train(X_train, X_test)

    return (X_train, X_test, y_train, y_test, predictions)


def get_svdpp():
    df = pd.read_csv("../data/time_location_aware.csv")
    df = df.sample(frac=0.001, random_state=0)

    (trainset, testset, predictions, dusers, ditems) = time_location_model(df)
    df_train, df_test, df_pred = get_tr_te_pr_time_loc(trainset, testset, predictions,
                                              dusers, ditems)

    X_train = pandas_to_spark(df_train)
    X_test = pandas_to_spark(df_test)
    predictions = pandas_to_spark(df_pred)

    X_train, X_test, y_train, y_test = create_test_train(X_train, X_test)

    return (X_train, X_test, y_train, y_test, predictions)


def get_dl():
    df = get_small_sample_df()

    X_train, X_test, predictions = DL_Model(df, df, None)

    X_train, X_test, y_train, y_test = create_test_train(X_train, X_test)

    return (X_train, X_test, y_train, y_test, predictions)


def get_content(ngram_range=(1, 1), sublinear_tf=True):
    df = get_small_sample_df()

    (df_train, df_test, df_pred) = content_based(df, ngram_range, sublinear_tf)

    X_train = pandas_to_spark(df_train)
    X_test = pandas_to_spark(df_test)
    predictions = pandas_to_spark(df_pred)

    X_train, X_test, y_train, y_test = create_test_train(X_train, X_test)

    return (X_train, X_test, y_train, y_test, predictions)


def get_metrics(X_train, X_test, y_train, y_test, predictions, name,
                metrics_dict):
    # print(f'Metrics for {name} ------------------------------------------')

    predictions = predictions.drop('rating')

    avg_overall_novelty, avg_novelty = calculate_novelty(
        X_train, X_test, predictions, sqlCtx)

    avg_overall_serendipity, avg_serendipity = calculate_serendipity(
        X_train, X_test, predictions, sqlCtx, rel_filter=1)

    rmse = calculate_rmse_using_rdd(X_train, X_test, predictions)
        
    coverage_rating = 1
    coverage = calculate_coverage(predictions, coverage_rating)

    # # pred_coverage = calculate_prediction_coverage(y_test, predictions)

    metrics_dict[name] = {}
    metrics_dict[name]['avg_overall_novelty'] = avg_overall_novelty
    metrics_dict[name]['avg_novelty'] = avg_novelty
    metrics_dict[name]['avg_overall_serendipity'] = avg_overall_serendipity
    metrics_dict[name]['avg_serendipity'] = avg_serendipity
    metrics_dict[name]['rmse'] = rmse
    print("Now running coverage")
    metrics_dict[name]['coverage'] = coverage

    print(f'avg_overall_novelty     = {avg_overall_novelty:.2f}')
    print(f'avg_novelty             = {avg_novelty:.2f}')
    print(f'avg_overall_serendipity = {avg_overall_serendipity:.2f}')
    print(f'avg_serendipity         = {avg_serendipity:.2f}')
    print(f'rmse                    = {rmse:.2f}')
    print(f'coverage                = {coverage:.6f}')

    # print(f"""{avg_overall_novelty} = avg_overall_novelty,
    #           {avg_novelty}=avg_novelty,

    #           """)


def param_tune_content(metrics_dict, ngram_range_list=[(1, 1),(1,5),
                                                       (1, 10)]):

    for ngram_range in ngram_range_list:
        for sublinear_tf in [True, False]:
            (X_train, X_test, y_train, y_test,
             predictions) = get_content(ngram_range=ngram_range,
                                        sublinear_tf=sublinear_tf)
            get_metrics(X_train, X_test, y_train, y_test, predictions,
                        f'ngram_range = {ngram_range}, sublinear_tf={sublinear_tf}', 
                        metrics_dict)

            # (X_train, X_test, y_train, y_test,
            #  predictions) = get_content(ngram_range=(1, 1),
            #                             sublinear_tf=sublinear_tf)
            # get_metrics(X_train, X_test, y_train, y_test, predictions,
            #             f'sublinear_tf = {sublinear_tf}', metrics_dict)
    
    
    

def load_metrics_cache(use_cache=True):
    cache_path = os.path.join(CACHE_PATH, f'metrics.msgpack')

    if use_cache and os.path.exists(cache_path):
        print(f'Loading from {cache_path}')
        metrics_df = pd.read_msgpack(cache_path)
        print(f'Loaded from {cache_path}')
    else:
        metrics_dict = dict()

        (X_train, X_test, y_train, y_test, predictions) = get_bias()
        get_metrics(X_train, X_test, y_train, y_test, predictions, 'BASELINE',
                    metrics_dict)

        print("BASELINE done")

        (X_train, X_test, y_train, y_test, predictions) = get_als()
        get_metrics(X_train, X_test, y_train, y_test, predictions,
                    'COLLABORATIVE FILTERING', metrics_dict)

        print("COLLABORATIVE done")

        (X_train, X_test, y_train, y_test, predictions) = get_content()
        get_metrics(X_train, X_test, y_train, y_test, predictions, 'CONTENT',
                    metrics_dict)

        print("CONTENT DONE")

        (X_train, X_test, y_train, y_test, predictions) = get_dl()
        get_metrics(X_train, X_test, y_train, y_test, predictions, 'DL',
                    metrics_dict)

        # metrics_df = pd.DataFrame(metrics_dict).T

        print("DL DONE")
        
        (X_train, X_test, y_train, y_test, predictions) = get_svdpp()
        get_metrics(X_train, X_test, y_train, y_test, predictions, 'SVDPP',
                    metrics_dict)

        # metrics_df = pd.DataFrame(metrics_dict).T

        print("SVDPP DONE")

        # pd.to_msgpack(cache_path, metrics_df)
        # print(f"Dumping to {cache_path}")

    return metrics_dict


def calculate_coverage(preds, rate):
    preds_pd = preds.toPandas()
    
    # breakpoint()		

    # Filter preds > 2
    preds_pd = preds_pd[preds_pd.prediction >= rate]
    
    # Count unique
    count_un = len(list(preds_pd['business_id'].unique()))
    
    # total = count_unique_biz_total
	
    total = 5873

    return count_un / total


if __name__ == '__main__':
    # Create context
    sc, sqlCtx = get_create_context()

    # metrics_df = load_metrics_cache(use_cache=False)
    metrics_dict = {}
    metrics_dict = load_metrics_cache(use_cache=True)
    
    # 
    # param_tune_content(metrics_dict, ngram_range_list=[(1, 1), (1, 2), (1, 3)])
    # print('DF IS')
    metrics_df = pd.DataFrame(metrics_dict).T
    print(metrics_df)
    metrics_df.to_csv('./Metrics_tuning_new.csv', index=True)