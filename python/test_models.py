# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 03:20:02 2019

@author: aksmi
"""

#import findspark
#findspark.init()

from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode
from pyspark.sql import SQLContext

from time import time
from common import CACHE_PATH, EXCEL_PATH
from common import load_pandas, pandas_to_spark

import scipy.sparse as sp
import numpy as np

import gc
from hermes import calculate_serendipity

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import numpy as np
from sklearn.model_selection import train_test_split

from colab_filtering_basline2 import get_als_model


def get_create_context():
    sc = SparkContext.getOrCreate()  # else get multiple contexts error
    sqlCtx = SQLContext(sc)

    return sc, sqlCtx


def create_test_train(train, test):
    x_cols = ['user_id', 'business_id', 'rating']
    y_cols = ['rating']

    X_train = train.select(x_cols)
    y_train = train.select(y_cols)

    X_test = train.select(x_cols)
    y_test = train.select(y_cols)

    return (X_train, X_test, y_train, y_test)


if __name__ == '__main__':

#    # Create context
#    sc, sqlCtx = get_create_context()
#    
#    # Get DF
#    frac = 0.001
#    df, _, _ = load_pandas()
#    print('Getting df')
#    df = df.sample(frac=frac, random_state=0)
#    print('Got df')
#    
#    start = time()
#    
#    # Get predictions from ALS
#    (y_predicted, model, rmse_train, rmse_test, coverage_train, coverage_test,
#     running_time, train, test) = get_als_model(df, 5)
#    
#    #         Get train test
#    X_train, X_test, y_train, y_test = create_test_train(train, test)
#
#    predictions = y_predicted.select(['user_id', 'business_id', 'prediction'])

    # Calculate metrics
    average_overall_serendipity, average_serendipity = calculate_serendipity(
        X_train, X_test, y_predicted, sqlCtx, rel_filter=1)
