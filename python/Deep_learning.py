import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import keras
import json
from tqdm.auto import tqdm
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import gc

try:
    from common import pandas_to_spark
except:
    from python.common import pandas_to_spark


def DL_Model(train, test=None, flag=None, plot=False):

    if flag == None:
        train = train[['user_id', 'business_id', 'rating']]
        test = test[['user_id', 'business_id', 'rating']]
        X = train[['user_id', 'business_id']]
        y = train['rating']

    elif flag == "Business":
        train = train[[
            'user_id', 'business_id', 'longitude', 'latitude', 'stars',
            'review_count', 'is_open', 'rating'
        ]]
        test = test[[
            'user_id', 'business_id', 'longitude', 'latitude', 'stars',
            'review_count', 'is_open', 'rating'
        ]]
        X = train[[
            'user_id', 'business_id', 'longitude', 'latitude', 'stars',
            'review_count', 'is_open'
        ]]
        y = train['rating']

    elif flag == "Both":
        train = train.drop(columns=[
            "date", "name_x", "address", "city", "state", "postal_code",
            "text", "name_y", "elite", "yelping_since"
        ],
                           axis=1)
        test = test.drop(columns=[
            "date", "name_x", "address", "city", "state", "postal_code",
            "text", "name_y", "elite", "yelping_since"
        ],
                         axis=1)
        y = train["rating"]
        X = train.drop(columns=["rating"], axis=1)

    le1 = LabelEncoder()
    le2 = LabelEncoder()
    X.fillna(0, inplace=True)
    X['user_id'] = le1.fit_transform(X['user_id'])
    X['business_id'] = le2.fit_transform(X['business_id'])

    model = KerasClassifier(build_fn=create_model, verbose=0)
    # define the grid search parameters
    optimizer = ['SGD', 'RMSprop', 'Adam']
    learn_rate = [0.001, 0.001, 0.1, 1]
    # param_grid = {
    #     'epochs': [1, 3, 6],
    #     "batch_size": [512, 1024, 2048]
    # }  

    param_grid = {
        'epochs': [1],
        "batch_size": [512]
    }
    #dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer)
    # grid = RandomizedSearchCV(estimator=model,
    #                           param_grid=param_grid,
    #                           n_jobs=1,
    #                           cv=3,
    #                           verbose=3)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2,verbose=3)
    grid_result = grid.fit(X, y)
    # summarize results
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    scores_df = pd.DataFrame(grid_result.cv_results_)
    scores_df.to_csv('DL_results.csv', index=False)
    if plot:
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    test['user_id'] = le1.transform(test['user_id'])
    test['business_id'] = le2.transform(test['business_id'])
    preds = test[['user_id', 'business_id', 'rating']]
    preds['prediction'] = np.clip(grid.best_estimator_.predict(
        test.drop('rating', axis=1)),
                                   a_min=1,
                                   a_max=5)
    #print(preds.head())

    train['user_id'] = le1.transform(train['user_id'])
    train['business_id'] = le2.transform(train['business_id'])
    train_ret = train[['user_id', 'business_id', 'rating']]
    #train_ret['predictions'] = np.clip(model.predict(train.drop('rating', axis=1)), a_min=1,a_max=5)
    #print(train_ret.head())

    preds_spark = pandas_to_spark(preds)
    train_spark = pandas_to_spark(train_ret)
    test_spark = pandas_to_spark(preds[['user_id', 'business_id', 'rating']])

    #print(train_spark)
    #print(test_spark)
    #print(preds_spark)

    del preds, train_ret,X,y, model
    gc.collect()

    return (train_spark, test_spark, preds_spark)


def test_out_of_sample_data():

    test_file_names = [
        'test_least_items', 'test_most_items', 'test_mostrecent',
        'test_nonprolific_users', 'test_prolific_users'
    ]

    train_df = pd.read_csv('../data/special/train.csv')

    X = train_df

    for test_file in test_file_names:
        test_df = pd.read_csv(f'../data/special/{test_file}.csv')

        X = pd.concat([X, test_df])

    le1 = LabelEncoder()
    le2 = LabelEncoder()

    le1.fit(X['user_id'])
    le2.fit(X['business_id'])

    X_train = train_df[['user_id', 'business_id']]
    y_train = train_df['rating']

    X_train['user_id'] = le1.transform(X_train['user_id'])
    X_train['business_id'] = le2.transform(X_train['business_id'])

    model = create_model()
    model.fit(X_train, y_train, batch_size=512, epochs=3)

    results = list()
    number_of_data_points = list()
    for test_file in test_file_names:
        test_df = pd.read_csv(f'../data/special/{test_file}.csv')

        X_test = test_df[['user_id', 'business_id']]
        y_test = test_df['rating']

        #print(train_df.shape)
        #print(test_df.shape[0])

        number_of_data_points.append(test_df.shape[0])

        X_test['user_id'] = le1.transform(X_test['user_id'])
        X_test['business_id'] = le2.transform(X_test['business_id'])

        preds = np.clip(model.predict(X_test), a_min=1, a_max=5)
        err = mean_squared_error(y_true=y_test, y_pred=preds)
        results.append(err)

        #print(f'{test_file} : {err}')
    results_df = pd.DataFrame(
        [test_file_names, results, number_of_data_points]).T
    results_df.columns = ['file_name', 'MSE', 'number_of_data_points']
    print(results_df)
    results_df.to_csv('./dl_diffrent_test_files.csv', index=False)


def create_model():
    # create model
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, input_dim=25, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation=None))

    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])
    #hist = model.fit(X,y, batch_size=1024,epochs=1, validation_split=0.3)

    return model


if __name__ == '__main__':
    warnings.simplefilter('ignore')
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    np.random.seed(42)
    tf.compat.v2.random.set_seed(42)
    
    # For just user id business id
    #df = pd.read_csv('../data/ratings.csv')
    
    # For all data
    df = pd.read_csv('../data/Allcombineddata.csv')
    
    DL_Model(df,df,"Both")


    #test_out_of_sample_data()
