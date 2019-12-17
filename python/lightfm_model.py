# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:54:27 2019

@author: aksmi
"""
try:
    from common import pandas_to_spark, spark_to_sparse, load_pandas
except:
    from python.common import pandas_to_spark, spark_to_sparse, load_pandas

from lightfm.cross_validation import random_train_test_split
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def lightfm_model(data, prec_at_k=100, train_split=0.8, epochs=10):
    """
        Code to evaluate LightFm model
        Data is a scipy sparse matrix
        
        https://arxiv.org/abs/1507.08439
    """
    model = LightFM(learning_rate=0.05, loss='logistic')

    train, test = random_train_test_split(data,
                                          test_percentage=1 - train_split)

    model.fit(train, epochs=epochs)  #, num_threads=1)

    train_precision = precision_at_k(model, train, k=prec_at_k)
    test_precision = precision_at_k(model,
                                    test,
                                    k=prec_at_k,
                                    train_interactions=train)

    train_auc = auc_score(model, train)
    test_auc = auc_score(model, test, train_interactions=train)

    print('Performance of LightFm Model \n')
    print(
        f'Precision \t Train: {train_precision.mean():.2f} \t Test: {test_precision.mean():.2f}'
    )
    print(
        f'AUC \t\t Train: {train_auc.mean():.2f} \t Test: {test_auc.mean():.2f}'
    )

    return (train_auc, test_auc, train_precision, test_precision, prec_at_k)


def tune_epochs(data, prec_at_k=100, train_split=0.8):

    train_auc_result, test_auc_result, train_precision_result, test_precision_result, prec_at_k_result = list(
    ), list(), list(), list(), list()
    train_auc_result_std, test_auc_result_std = list(), list()
    epochs = [i for i in range(1, 11)]
    for epoch in tqdm(epochs):
        (train_auc, test_auc, train_precision, test_precision,
         prec_at_k) = lightfm_model(data=sparse_mat,
                                    prec_at_k=prec_at_k,
                                    train_split=train_split,
                                    epochs=epoch)
        train_auc_result.append(train_auc.mean())
        test_auc_result.append(test_auc.mean())
        train_precision_result.append(train_precision.mean())
        test_precision_result.append(test_precision.mean())
        prec_at_k_result.append(prec_at_k)
        train_auc_result_std.append(train_auc.std())
        test_auc_result_std.append(test_auc.std())

    results_df = pd.DataFrame([
        epochs, train_auc_result, test_auc_result, train_auc_result_std,
        test_auc_result_std, train_precision_result, test_precision_result,
        prec_at_k_result
    ]).T
    results_df.columns = [
        'epochs', 'train_auc_mean', 'test_auc_mean', 'train_auc_std',
        'test_auc_std', 'train_precision', 'test_precision', 'prec_at_k'
    ]
    results_df.to_csv('./LightFM_epochs_tune.csv', index=False)


def plot_metrics(train_auc, test_auc, train_precision, test_precision,
                 prec_at_k):

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].hist(train_auc, bins='auto')
    ax[0, 0].title.set_text('Distribution of Train AUC score over users')
    ax[0, 0].set_ylabel('Count')
    ax[0, 0].set_xlabel('AUC Score')

    ax[0, 1].hist(test_auc, bins='auto')
    ax[0, 1].title.set_text('Distribution of Test AUC score over users')
    ax[0, 1].set_ylabel('Count')
    ax[0, 1].set_xlabel('AUC Score')

    ax[1, 0].hist(train_precision, bins='auto')
    ax[1, 0].title.set_text(
        f'Distribution of Train Precision @ {prec_at_k} for all users')
    ax[1, 0].set_ylabel('Count')
    ax[1, 0].set_xlabel(f'Precision @ {prec_at_k}')

    ax[1, 1].hist(test_precision, bins='auto')
    ax[1, 1].title.set_text(
        f'Distribution of Test Precision @ {prec_at_k} for all users')
    ax[1, 1].set_ylabel('Count')
    ax[1, 1].set_xlabel(f'Precision @ {prec_at_k}')

    plt.show()

    print('\n')


if __name__ == '__main__':
    np.random.seed(42)
    frac = 0.01
    df, _, _ = load_pandas()
    print('Getting df')
    df = df.sample(frac=frac, random_state=0)
    spark_df = pandas_to_spark(df)

    sparse_mat = spark_to_sparse(spark_df)

    # breakpoint()

    # To tune epochs, uncomment the following line
    #tune_epochs(data=sparse_mat, prec_at_k=100, train_split=0.8)

    (train_auc, test_auc, train_precision, test_precision,
     prec_at_k) = lightfm_model(sparse_mat, prec_at_k=100, train_split=0.8)

    plot_metrics(train_auc, test_auc, train_precision, test_precision,
                 prec_at_k)
