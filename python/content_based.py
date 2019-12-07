import findspark
findspark.init()
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode
from time import time
from common import CACHE_PATH, EXCEL_PATH
from common import load_pandas, pandas_to_spark
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def tfidf_vectorizer(df):
    # https://stackoverflow.com/questions/45981037/sklearn-tf-idf-to-drop-numbers
    args = {
        'lowercase': True,
        'stop_words': 'english',
        'token_pattern': u'(?ui)\\b\\w*[a-z]+\\w*\\b'
    }
    reviews = df['text'].values
    print('converted to values')
    vectorizer = TfidfVectorizer(**args)
    print('converting to review vector')
    df['review_vector'] = list(vectorizer.fit_transform(reviews))
    # breakpoint()
    print('converted to review vector')


def get_avg_vectors(df):
    df_agg_users = df.groupby('user_id')['review_vector'].sum() / \
                   df.groupby('user_id')['user_id'].count()
    df_agg_rest = df.groupby('business_id')['review_vector'].sum() / \
                  df.groupby('business_id')['user_id'].count()
    return df_agg_users, df_agg_rest


def get_recommendations(user_mat, item_mat):
    # multiply matrices to get ratings
    cosine_similarities = user_mat * item_mat.T
    return cosine_similarities


if __name__ == '__main__':
    frac = 0.01
    df, _, _ = load_pandas()
    print('Getting df')
    df = df.sample(frac=frac, random_state=0)
    print('Got df')
    start = time()
    tfidf_vectorizer(df)
    running_time = time() - start
    print(f'running time = {running_time}')
    print('Grouping')
    df_agg_users, df_agg_rest = get_avg_vectors(df)
    users_mat = sp.vstack(df_agg_users)
    rest_mat = sp.vstack(df_agg_rest)

    # TODO check multiplication
    recos = get_recommendations(users_mat, rest_mat)
    top_ten_indicies = []
    for row_num in range(recos.shape[0]):
        if row_num % 1000 == 0:
            print(row_num)
        row = recos.getrow(row_num).toarray()[0].ravel()
        top_ten_indicies.append(row.argsort()[-3:])
        # top_ten_values = row[row.argsort()[-10:]]