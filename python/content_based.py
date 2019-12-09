from common import CACHE_PATH, EXCEL_PATH

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode
from time import time

from common import load_pandas, pandas_to_spark

import scipy.sparse as sp
import numpy as np

import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def tfidf_vectorizer(df):
    # https://stackoverflow.com/questions/45981037/sklearn-tf-idf-to-drop-numbers
    kwargs = {
        'lowercase': True,
        'stop_words': 'english',
        'token_pattern': u'(?ui)\\b\\w*[a-z]+\\w*\\b'
    }
    reviews = df['text'].values
    print('converted to values')
    vectorizer = TfidfVectorizer(**kwargs)
    print('converting to review vector')
    df['review_vector'] = list(vectorizer.fit_transform(reviews))
    # breakpoint()
    print('converted to review vector')


def get_avg_vectors(df):
    df_agg_users = df.groupby('user_id')['review_vector'].sum() / \
                   df.groupby('user_id')['user_id'].count()
    df_agg_rest = df.groupby('business_id')['review_vector'].sum() / \
                  df.groupby('business_id')['user_id'].count()

    user_df = df_agg_users.reset_index()
    business_df = df_agg_rest.reset_index()
    user_map = dict(zip(user_df['user_id'].values, user_df['user_id'].index))
    business_map = dict(
        zip(business_df['business_id'].values,
            business_df['business_id'].index))

    return (df_agg_users, df_agg_rest, user_map, business_map)


def get_recommendations(user_mat, item_mat):
    # multiply matrices to get ratings
    cosine_similarities = user_mat * item_mat.T
    return cosine_similarities


def get_sparse_mask(df):
    rows, r_pos = np.unique(df['user_id'], return_inverse=True)
    cols, c_pos = np.unique(df['business_id'], return_inverse=True)
    s = sp.coo_matrix((np.ones(r_pos.shape, int), (r_pos, c_pos)))
    return s


def find_recos(df):
    df_agg_users, df_agg_rest, _, _ = get_avg_vectors(df)
    users_mat = sp.vstack(df_agg_users)
    rest_mat = sp.vstack(df_agg_rest)

    del df_agg_users
    del df_agg_rest

    #    sparse_mask = get_sparse_mask(df)

    # # TODO check multiplication
    recos = get_recommendations(users_mat, rest_mat)

    gc.collect()

    del users_mat
    del rest_mat

    #TODO recos = sparse_mask.multiply(recos)

    return recos


def get_top_n(recos, num_rec):
    num_rec = 3
    top_n_indicies = np.zeros(shape=(recos.shape[0], num_rec))
    for row_num in range(recos.shape[0]):
        if row_num % 10000 == 0:
            print(row_num)


#            break
        row = recos.getrow(row_num).toarray()[0].ravel()
        top_n_indicies[row_num] = row.argsort()[-num_rec:]
        return top_n_indicies


def get_tr_te_pr():
    frac = 0.001
    df, _, _ = load_pandas()
    print('Getting df')
    df = df.sample(frac=frac, random_state=0)
    print('Got df')

    tfidf_vectorizer(df)

    (df_agg_users, df_agg_rest, user_map, business_map) = get_avg_vectors(df)

    df_map = df.copy()
    df_map['business_id'] = df_map['business_id'].map(business_map)
    df_map['user_id'] = df_map['user_id'].map(user_map)

    df_map['business_id'] = df_map['business_id'].astype('int32')
    df_map['user_id'] = df_map['user_id'].astype('int32')

    start = time()

    gc.collect()

    running_time = time() - start
    print(f'running time = {running_time}')
    print('Grouping')

    recos = find_recos(df)

    recos_dense = recos.todense()

    df_map['prediction'] = None

    msk = np.random.rand(len(df_map)) < 0.8
    df_train = df_map[msk][['user_id', 'business_id', 'rating']]
    df_test = df_map[~msk][['user_id', 'business_id', 'rating']]

    for index, row in df_map.iterrows():
        df_map.at[index, 'prediction'] = recos_dense[row['user_id'],
                                                     row['business_id']]

    predictions = df_map[['user_id', 'business_id', 'rating', 'prediction']]

    predictions['prediction'] = predictions['prediction'].astype('float64')

    return (df_train, df_test, predictions)


if __name__ == '__main__':
    # frac = 0.001
    # df, _, _ = load_pandas()
    # print('Getting df')
    # df = df.sample(frac=frac, random_state=0)
    # print('Got df')

    # tfidf_vectorizer(df)

    # (df_agg_users, df_agg_rest, user_map, business_map) = get_avg_vectors(df)

    # df_map = df.copy()
    # df_map['business_id'] = df_map['business_id'].map(business_map)
    # df_map['user_id'] = df_map['user_id'].map(user_map)

    # df_map['business_id'] = df_map['business_id'].astype('int32')
    # df_map['user_id'] = df_map['user_id'].astype('int32')

    # start = time()

    # gc.collect()

    # running_time = time() - start
    # print(f'running time = {running_time}')
    # print('Grouping')

    # recos = find_recos(df)

    # recos_dense = recos.todense()

    # df_map['prediction'] = None

    # for index, row in df_map.iterrows():
    #     df_map.at[index, 'prediction'] = recos_dense[row['user_id'], row['business_id']]

    (df_train, df_test, predictions) = get_tr_te_pr()
