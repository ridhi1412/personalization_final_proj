# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 03:20:02 2019

@author: aksmi
"""

#import findspark
#findspark.init()
from common import CACHE_PATH, EXCEL_PATH


from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode
from pyspark.sql import SQLContext

from time import time
from common import load_pandas, pandas_to_spark

import scipy.sparse as sp
import numpy as np

import gc
from hermes import (calculate_serendipity, 
                    calculate_novelty,
                    calculate_novelty_bias,
                    calculate_prediction_coverage)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import numpy as np
from sklearn.model_selection import train_test_split

from colab_filtering_basline2 import get_als_model
from bias1 import baseline_bias_model, get_tr_te_pr
from content_based import get_tr_te_pr as get_tr_te_pr_content 


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



def get_als():
    frac = 0.001
    df, _, _ = load_pandas()
    print('Getting df')
    df = df.sample(frac=frac, random_state=0)
    print('Got df')

    # Get predictions from ALS
    (y_predicted, model, rmse_train, rmse_test, coverage_train, coverage_test,
     running_time, train, test) = get_als_model(df, 5)

#         Get train test
    X_train, X_test, y_train, y_test = create_test_train(train, test)
    
    predictions = y_predicted.select(['user_id', 'business_id', 'prediction'])
    
    return (X_train, X_test, y_train, y_test, predictions)


def get_bias():
    frac = 0.001
    df, _, _ = load_pandas()
    df = df.sample(frac=frac, random_state=0)
    
    (trainset, testset, predictions, dusers, ditems) = baseline_bias_model(df)
    df_train, df_test, df_pred = get_tr_te_pr(trainset, testset, predictions, dusers, ditems)
    
    X_train = pandas_to_spark(df_train)
    X_test = pandas_to_spark(df_test)
    predictions = pandas_to_spark(df_pred)
    
    X_train, X_test, y_train, y_test = create_test_train(X_train, X_test)
    
    return (X_train, X_test, y_train, y_test, predictions)

def get_content():
    df_train, df_test, df_pred = get_tr_te_pr_content()
    
    X_train = pandas_to_spark(df_train)
    X_test = pandas_to_spark(df_test)
    predictions = pandas_to_spark(df_pred)
    
    X_train, X_test, y_train, y_test = create_test_train(X_train, X_test)
    
    return (X_train, X_test, y_train, y_test, predictions)

def get_metrics(X_train, X_test, y_train, y_test, predictions, name):
    print(f'Metrics for {name} ------------------------------------------')
    avg_overall_novelty, avg_novelty = calculate_novelty(
        X_train, X_test, predictions, sqlCtx)
    
    
    avg_overall_serendipity, avg_serendipity = calculate_serendipity(
        X_train, X_test, predictions, sqlCtx, rel_filter=1)
    
    
    # # pred_coverage = calculate_prediction_coverage(y_test, predictions)
    
    print(f"""{avg_overall_novelty} = avg_overall_novelty, 
              {avg_novelty}=avg_novelty, 
              {avg_overall_serendipity}=avg_overall_serendipity,
              {avg_serendipity} = avg_serendipity,
              
              """)
              
    # print(f"""{avg_overall_novelty} = avg_overall_novelty, 
    #           {avg_novelty}=avg_novelty, 
              
    #           """)
              
    print(f"""{avg_overall_serendipity} = avg_overall_serendipity, 
     {avg_serendipity}=avg_serendipity, 
     
     """)
     


if __name__ == '__main__':

    # Create context
    sc, sqlCtx = get_create_context()
    
    # (X_train, X_test, y_train, y_test, predictions) = get_bias()
    # get_metrics(X_train, X_test, y_train, y_test, predictions, 'BASELINE')
    
    # print("BASELINE done")
    
    
    # (X_train, X_test, y_train, y_test, predictions) = get_als()
    # get_metrics(X_train, X_test, y_train, y_test, predictions, 'COLLABORATIVE FILTERING')

    # print("COLLABORATIVE done")

    (X_train, X_test, y_train, y_test, predictions) = get_content()
    get_metrics(X_train, X_test, y_train, y_test, predictions, 'CONTENT')
    
    print("CONTENT DONE")
    
    
    
    
