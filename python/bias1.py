# <<<<<<< HEAD
import pandas as pd
import numpy as np
import surprise
from surprise import Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import BaselineOnly

#import os
try:
    from common import CACHE_PATH, EXCEL_PATH, load_pandas, pandas_to_spark
except:
    from python.common import (CACHE_PATH, EXCEL_PATH, load_pandas,
                               pandas_to_spark)


def baseline_bias_model(df):
    """
        Shows the performance of model based on just bias
    """
    ratings_pandas_df = df.drop(columns=['date', 'text'])
    #    ratings_pandas_df.columns = ['user_id', 'business_id', 'rating']

    reader = Reader(rating_scale=(1, 5))  #TODO figure out

    data = surprise.dataset.Dataset.load_from_df(df=ratings_pandas_df,
                                                 reader=reader)

    ts = data.build_full_trainset()
    dusers = ts._raw2inner_id_users
    ditems = ts._raw2inner_id_items

    trainset, testset = train_test_split(data)

    algo = BaselineOnly()
    algo.fit(trainset)

    # testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    print('\n')
    return (trainset, testset, predictions, dusers, ditems)


def get_tr_te_pr(train, test, preds, dusers, ditems):
    iterator = train.all_ratings()
    df_train = pd.DataFrame(columns=['user_id', 'business_id', 'rating'])
    # df_train = pd.DataFrame(columns=['user_id', 'business_id', 'rating'])

    i = 0
    for (uid, iid, rating) in iterator:
        df_train.loc[i] = [uid, iid, rating]
        i = i + 1

    d1 = train._raw2inner_id_users
    d1 = {value: key for (key, value) in d1.items()}
    # df_train['user_id'] = df_train['user_id'].map(d1)

    d2 = train._raw2inner_id_items
    d2 = {value: key for (key, value) in d2.items()}
    # df_train['business_id'] = df_train['business_id'].map(d2)

    # breakpoint()
    df_test = pd.DataFrame.from_records(
        test, columns=['user_id', 'business_id', 'rating'])
    df_test['business_id'] = df_test['business_id'].map(ditems)
    df_test['user_id'] = df_test['user_id'].map(dusers)

    df_pred = pd.DataFrame.from_records(preds,
                                        columns=[
                                            'user_id', 'business_id', 'rating',
                                            'prediction', 'isimpossible'
                                        ])

    df_pred['business_id'] = df_pred['business_id'].map(ditems)
    df_pred['user_id'] = df_pred['user_id'].map(dusers)

    df_train['business_id'] = df_train['business_id'].astype('int32')
    df_train['user_id'] = df_train['user_id'].astype('int32')

    df_test['business_id'] = df_test['business_id'].astype('int32')
    df_test['user_id'] = df_test['user_id'].astype('int32')

    df_pred['business_id'] = df_pred['business_id'].astype('int32')
    df_pred['user_id'] = df_pred['user_id'].astype('int32')

    df_pred.drop(columns=['isimpossible'], inplace=True)

    return (df_train, df_test, df_pred)


if __name__ == '__main__':
    frac = 0.001
    df, _, _ = load_pandas()
    df = df.sample(frac=frac, random_state=0)
    (trainset, testset, predictions, dusers, ditems) = baseline_bias_model(df)
    df_train, df_test, df_pred = get_tr_te_pr(trainset, testset, predictions,
                                              dusers, ditems)

    spark_train = pandas_to_spark(df_train)
    spark_test = pandas_to_spark(df_test)
    spark_pred = pandas_to_spark(df_pred)

# =======
# import pandas as pd
# import numpy as np
# import surprise
# from surprise import Reader
# from surprise.model_selection import cross_validate, train_test_split
# from surprise import BaselineOnly

# #import os
# from common import CACHE_PATH, EXCEL_PATH, load_pandas, pandas_to_spark

# def baseline_bias_model(df):
#     """
#         Shows the performance of model based on just bias
#     """
#     ratings_pandas_df = df.drop(columns=['date', 'text'])
#     #    ratings_pandas_df.columns = ['user_id', 'business_id', 'rating']

#     reader = Reader(rating_scale=(1, 5))  #TODO figure out

#     data = surprise.dataset.Dataset.load_from_df(df=ratings_pandas_df,
#                                                  reader=reader)

#     ts = data.build_full_trainset()
#     dusers = ts._raw2inner_id_users
#     ditems = ts._raw2inner_id_items

#     # breakpoint()
#     trainset, testset = train_test_split(data)

#     algo = BaselineOnly()
#     algo.fit(trainset)

#     # testset = trainset.build_anti_testset()
#     predictions = algo.test(testset)

#     print('\n')
#     return (trainset, testset, predictions, dusers, ditems)

# def get_tr_te_pr(train, test, preds, dusers, ditems):
#     iterator = train.all_ratings()
#     df_train = pd.DataFrame(columns=['user_id', 'business_id', 'rating'])
#     # df_train = pd.DataFrame(columns=['user_id', 'business_id', 'rating'])

#     i = 0
#     for (uid, iid, rating) in iterator:
#         df_train.loc[i] = [uid, iid, rating]
#         i = i+1

#     d1 = train._raw2inner_id_users
#     d1 = {value:key for (key, value) in d1.items()}
#     # df_train['user_id'] = df_train['user_id'].map(d1)

#     d2 = train._raw2inner_id_items
#     d2 = {value:key for (key, value) in d2.items()}
#     # df_train['business_id'] = df_train['business_id'].map(d2)

#     # breakpoint()
#     df_test = pd.DataFrame.from_records(test, columns=['user_id', 'business_id', 'rating'])
#     df_test['business_id'] = df_test['business_id'].map(ditems)
#     df_test['user_id'] = df_test['user_id'].map(dusers)

#     df_pred = pd.DataFrame.from_records(preds, columns=['user_id', 'business_id', 'rating',
#                                                         'prediction', 'isimpossible'])

#     df_pred['business_id'] = df_pred['business_id'].map(ditems)
#     df_pred['user_id'] = df_pred['user_id'].map(dusers)

#     df_train['business_id'] = df_train['business_id'].astype('int32')
#     df_train['user_id'] = df_train['user_id'].astype('int32')

#     df_test['business_id'] = df_test['business_id'].astype('int32')
#     df_test['user_id'] = df_test['user_id'].astype('int32')

#     df_pred['business_id'] = df_pred['business_id'].astype('int32')
#     df_pred['user_id'] = df_pred['user_id'].astype('int32')

#     df_pred.drop(columns=['isimpossible'], inplace=True)

#     return (df_train, df_test, df_pred)

# if __name__ == '__main__':
#     frac = 0.001
#     df, _, _ = load_pandas()
#     df = df.sample(frac=frac, random_state=0)
#     (trainset, testset, predictions, dusers, ditems) = baseline_bias_model(df)
#     df_train, df_test, df_pred = get_tr_te_pr(trainset, testset, predictions, dusers, ditems)

#     spark_train = pandas_to_spark(df_train)
#     spark_test = pandas_to_spark(df_test)
#     spark_pred = pandas_to_spark(df_pred)

# >>>>>>> origin
