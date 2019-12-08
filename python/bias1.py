import pandas as pd
import numpy as np
import surprise
from surprise import Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import BaselineOnly

#import os
from common import CACHE_PATH, EXCEL_PATH, load_pandas


def baseline_bias_model(df):
    """
        Shows the performance of model based on just bias
    """
    ratings_pandas_df = df.drop(columns=['date', 'text'])
    #    ratings_pandas_df.columns = ['user_id', 'business_id', 'rating']
    
    reader = Reader(rating_scale=(1, 5))  #TODO figure out

    data = surprise.dataset.Dataset.load_from_df(df=ratings_pandas_df,
                                                 reader=reader)
    
    # breakpoint()    
    trainset, testset = train_test_split(data)
    
    algo = BaselineOnly()
    algo.fit(trainset)

    # testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
        
    
#    _ = cross_validate(
#        surprise.prediction_algorithms.baseline_only.BaselineOnly(),
#        data,
#        measures=['RMSE', 'MAE'],
#        cv=5,
#        verbose=1,
#        n_jobs=-1)

    print('\n')
    return (trainset, testset, predictions)


if __name__ == '__main__':
    frac = 0.001
    df, _, _ = load_pandas()
    df = df.sample(frac=frac, random_state=0)
    train, test, preds = baseline_bias_model(df)
    
    iterator = train.all_ratings()
    df_train = pd.DataFrame(columns=['user_id', 'business_id', 'rating'])
    df_train = pd.DataFrame(columns=['user_id', 'business_id', 'rating'])
    i = 0
    for (uid, iid, rating) in iterator:
        df_train.loc[i] = [uid, iid, rating]
        i = i+1
    
    df_train['user_id'] = df_train['user_id'].map(train._inner2raw_id_users)
    df_test['business_id'] = df_test['business_id'].map(train._inner2raw_id_items)
    
    df_test = pd.DataFrame.from_records(test, columns=['user_id', 'business_id', 'rating'])
        
    
    