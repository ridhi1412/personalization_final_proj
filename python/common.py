# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:05:32 2019

@author: rmahajan14
"""
import os
import pandas as pd

import getpass
user = getpass.getuser()

if user == 'aksmi':
    import findspark
    findspark.init()

from pyspark import SparkContext
from pyspark.sql import SQLContext
import numpy as np

from scipy.sparse import csr_matrix

import sys
from sys import platform

if user == 'rmahajan14':
    DIRPATH = r'P:\rmahajan14\columbia\fall 2019\Personalization\final_project'
if user == 'anirudh':
    DIRPATH = r'../'
if user == 'Sheetal':
    DIRPATH = r'../'
if user == 'aksmi':
    DIRPATH = r'E:\yelp'

EXCEL_PATH = os.path.join(DIRPATH, 'data')
CACHE_PATH = os.path.join(DIRPATH, 'cache')

import json
from tqdm.auto import tqdm

FRACTION = 0.001


def pandas_to_spark(pandas_df):
    sc = SparkContext.getOrCreate()  # else get multiple contexts error
    sql_sc = SQLContext(sc)
    spark_df = sql_sc.createDataFrame(pandas_df)
    return spark_df


def load_pandas(file_name='review.json', use_cache=True):
    cache_path = os.path.join(CACHE_PATH, f'load_pandas.msgpack')
    if use_cache and os.path.exists(cache_path):
        print(f'Loading from {cache_path}')
        ratings, user_counts, active_users = pd.read_msgpack(cache_path)
        print(f'Loaded from {cache_path}')
    else:
        line_count = len(
            open(os.path.join(EXCEL_PATH, file_name),
                 encoding='utf8').readlines())
        user_ids, business_ids, stars, dates, text = [], [], [], [], []
        with open(os.path.join(EXCEL_PATH, file_name), encoding='utf8') as f:
            for line in tqdm(f, total=line_count):
                blob = json.loads(line)
                user_ids += [blob["user_id"]]
                business_ids += [blob["business_id"]]
                stars += [blob["stars"]]
                dates += [blob["date"]]
                text += [blob["text"]]

        ratings = pd.DataFrame({
            "user_id": user_ids,
            "business_id": business_ids,
            "rating": stars,
            "text": text,
            "date": dates
        })
        user_counts = ratings["user_id"].value_counts()
        active_users = user_counts.loc[user_counts >= 5].index.tolist()

        pd.to_msgpack(cache_path, (ratings, user_counts, active_users))
        print(f'Dumping to {cache_path}')
    return ratings, user_counts, active_users


def location_and_time_data(file_name='location_time', use_cache=True):
    """
        Returns location and time discounted rating
    """
    cache_path = os.path.join(CACHE_PATH, f'location_and_time_data.msgpack')
    if use_cache and os.path.exists(cache_path):
        print(f'Loading from {cache_path}')
        detail_df = pd.read_msgpack(cache_path)
        print(f'Loaded from {cache_path}')
    else:
        # To get review json
        line_count = len(
            open(os.path.join(EXCEL_PATH, "review.json"),
                 encoding='utf8').readlines())
        user_ids, business_ids, stars, dates = [], [], [], []
        with open(os.path.join(EXCEL_PATH, "review.json"),
                  encoding='utf8') as f:
            for line in tqdm(f, total=line_count):
                blob = json.loads(line)
                user_ids += [blob["user_id"]]
                business_ids += [blob["business_id"]]
                stars += [blob["stars"]]
                dates += [blob["date"]]

        ratings = pd.DataFrame({
            "user_id": user_ids,
            "business_id": business_ids,
            "rating": stars,
            "date": dates
        })

        line_count = len(
            open(os.path.join(EXCEL_PATH, "business.json"),
                 encoding='utf8').readlines())
        name, business_id, address, city, state, postal_code = [], [], [], [], [], []
        latitude, longitude, stars, review_count = [], [], [], []
        is_open, attributes, GoodForKids, categories, hours = [], [], [], [], []

        with open(os.path.join(EXCEL_PATH, "business.json"),
                  encoding='utf8') as f:
            for line in tqdm(f, total=line_count):
                blob = json.loads(line)
                name += [blob["name"]]
                business_id += [blob["business_id"]]
                address += [blob["address"]]
                city += [blob["city"]]
                state += [blob["state"]]
                postal_code += [blob["postal_code"]]
                latitude += [blob["latitude"]]
                longitude += [blob["longitude"]]
                stars += [blob["stars"]]
                review_count += [blob["review_count"]]
                is_open += [blob["is_open"]]

        business = pd.DataFrame({
            "name": name,
            "business_id": business_id,
            "address": address,
            "city": city,
            "state": state,
            "postal_code": postal_code,
            "latitude": latitude,
            "longitude": longitude,
            'stars': stars,
            'review_count': review_count,
            'is_open': is_open
        })

        detail_df = pd.merge(left=ratings,
                             right=business,
                             on='business_id',
                             how='left')
        mean_lat = detail_df.groupby(
            'user_id')['latitude'].mean().reset_index()
        mean_long = detail_df.groupby(
            'user_id')['longitude'].mean().reset_index()
        mean_df = pd.merge(mean_lat, mean_long, on='user_id')
        mean_df.columns = ['user_id', 'mean_lat', 'mean_long']
        detail_df = pd.merge(detail_df, mean_df, on='user_id', how='left')

        detail_df['distance'] = (
            (detail_df['mean_lat'] - detail_df['latitude'])**2) + (
                (detail_df['mean_long'] - detail_df['longitude'])**2)

        # For date distances
        detail_df['date'] = pd.to_datetime(detail_df['date'])
        last_date = detail_df.groupby('user_id')['date'].max().reset_index()
        last_date.columns = ['user_id', 'last_date']
        detail_df = pd.merge(detail_df, last_date, on='user_id', how='left')

        # Months instead of days
        detail_df['date_diff'] = (detail_df['last_date'] - detail_df['date'])
        detail_df['date_diff'] = detail_df['date_diff'].dt.days / 30

        # e(1/1+dist)/e
        # e(1/ log(date))/e
        detail_df['dist_scale'] = np.exp(
            1 / (1 + detail_df['distance'])) / (np.exp(1))
        detail_df['date_scale'] = np.exp(
            1 / (1 + np.log(detail_df['date_diff'] + 1))) / np.exp(1)

        # Multiplying rating scale with the dist
        detail_df[
            'date_rating'] = detail_df['rating'] * detail_df['date_scale']
        detail_df[
            'dist_rating'] = detail_df['rating'] * detail_df['dist_scale']
        detail_df['date_dist_rating'] = detail_df['date_scale'] * detail_df[
            'dist_scale'] * detail_df['rating']

        pd.to_msgpack(cache_path, (detail_df))
        print(f'Dumping to {cache_path}')
    return detail_df


def all_data(file_name='all_data', use_cache=True):
    """
        Returns business and user meta data with the ratings
    """
    cache_path = os.path.join(CACHE_PATH, f'all_data.msgpack')
    if use_cache and os.path.exists(cache_path):
        print(f'Loading from {cache_path}')
        temp = pd.read_msgpack(cache_path)
        print(f'Loaded from {cache_path}')
    else:
        # To get review json
        line_count = len(
            open(os.path.join(EXCEL_PATH, "review.json"),
                 encoding='utf8').readlines())
        user_ids, business_ids, stars, dates = [], [], [], []
        with open(os.path.join(EXCEL_PATH, "review.json"),
                  encoding='utf8') as f:
            for line in tqdm(f, total=line_count):
                blob = json.loads(line)
                user_ids += [blob["user_id"]]
                business_ids += [blob["business_id"]]
                stars += [blob["stars"]]
                dates += [blob["date"]]

        ratings = pd.DataFrame({
            "user_id": user_ids,
            "business_id": business_ids,
            "rating": stars,
            "date": dates
        })

        line_count = len(
            open(os.path.join(EXCEL_PATH, "business.json"),
                 encoding='utf8').readlines())
        name, business_id, address, city, state, postal_code = [], [], [], [], [], []
        latitude, longitude, stars, review_count = [], [], [], []
        is_open, attributes, GoodForKids, categories, hours = [], [], [], [], []

        with open(os.path.join(EXCEL_PATH, "business.json"),
                  encoding='utf8') as f:
            for line in tqdm(f, total=line_count):
                blob = json.loads(line)
                name += [blob["name"]]
                business_id += [blob["business_id"]]
                address += [blob["address"]]
                city += [blob["city"]]
                state += [blob["state"]]
                postal_code += [blob["postal_code"]]
                latitude += [blob["latitude"]]
                longitude += [blob["longitude"]]
                stars += [blob["stars"]]
                review_count += [blob["review_count"]]
                is_open += [blob["is_open"]]

        business = pd.DataFrame({
            "name": name,
            "business_id": business_id,
            "address": address,
            "city": city,
            "state": state,
            "postal_code": postal_code,
            "latitude": latitude,
            "longitude": longitude,
            'stars': stars,
            'review_count': review_count,
            'is_open': is_open
        })

        # To get user json

        line_count = len(
            open(os.path.join(EXCEL_PATH, "user.json"),
                 encoding='utf8').readlines())
        name, user_id, review_count, yelping_since, useful = [], [], [], [], []
        funny, cool, elite, fans = [], [], [], []
        average_stars, compliment_hot, compliment_more, compliment_profile = [], [], [], []
        compliment_cute, compliment_list, compliment_note, compliment_plain, compliment_cool = [], [], [], [], []
        compliment_funny, compliment_writer, compliment_photos = [], [], []

        with open(os.path.join(EXCEL_PATH, "user.json"), encoding='utf8') as f:
            for line in tqdm(f, total=line_count):
                blob = json.loads(line)
                name += [blob["name"]]
                user_id += [blob["user_id"]]
                review_count += [blob["review_count"]]
                yelping_since += [blob["yelping_since"]]
                useful += [blob["useful"]]
                funny += [blob["funny"]]
                cool += [blob["cool"]]
                elite += [blob["elite"]]
                fans += [blob["fans"]]
                average_stars += [blob["average_stars"]]
                compliment_hot += [blob["compliment_hot"]]
                compliment_more += [blob["compliment_more"]]
                compliment_profile += [blob["compliment_profile"]]
                compliment_cute += [blob["compliment_cute"]]
                compliment_list += [blob["compliment_list"]]
                compliment_note += [blob["compliment_note"]]
                compliment_plain += [blob["compliment_plain"]]
                compliment_cool += [blob["compliment_cool"]]
                compliment_funny += [blob["compliment_funny"]]
                compliment_writer += [blob["compliment_writer"]]
                compliment_photos += [blob["compliment_photos"]]

        user = pd.DataFrame({
            "name": name,
            "user_id": user_id,
            "review_count": review_count,
            "yelping_since": yelping_since,
            "useful": useful,
            "funny": funny,
            "cool": cool,
            "elite": elite,
            "fans": fans,
            "average_stars": average_stars,
            "compliment_hot": compliment_hot,
            "compliment_more": compliment_more,
            "compliment_profile": compliment_profile,
            "compliment_cute": compliment_cute,
            "compliment_list": compliment_list,
            "compliment_note": compliment_note,
            "compliment_plain": compliment_plain,
            "compliment_cool": compliment_cool,
            "compliment_funny": compliment_funny,
            "compliment_writer": compliment_writer,
            "compliment_photos": compliment_photos
        })

        # To get tip json

        line_count = len(
            open(os.path.join(EXCEL_PATH, "tip.json"),
                 encoding='utf8').readlines())
        business_id, user_id, text, date, compliment_count = [], [], [], [], []

        with open(os.path.join(EXCEL_PATH, "tip.json"), encoding='utf8') as f:
            for line in tqdm(f, total=line_count):
                blob = json.loads(line)
                business_id += [blob["business_id"]]
                user_id += [blob["user_id"]]
                text += [blob["text"]]
                date += [blob["date"]]
                compliment_count += [blob["compliment_count"]]

        tip = pd.DataFrame({
            "business_id": business_id,
            "user_id": user_id,
            "text": text,
            "date": date,
            "compliment_count": compliment_count
        })
        temp = pd.merge(ratings, business, on='business_id', how='left')
        temp = pd.merge(temp, user, on='user_id', how='left')
        temp = pd.merge(temp,
                        tip,
                        on=['business_id', 'user_id', 'date'],
                        how='left')

        pd.to_msgpack(cache_path, (temp))
        print(f'Dumping to {cache_path}')
    return temp


def spark_to_sparse(spark_df, user_or_item='user'):
    """
        Makes a spark data frame sparse for models such as nearest neighbors
        and LightFM
    """
    df = spark_df.drop('timestamp')
    pd_df = df.toPandas()

    row = pd_df['userId'].values
    column = pd_df['movieId'].values
    values = pd_df['rating'].values

    num_rows = max(pd_df['userId'])
    num_columns = max(pd_df['movieId'])

    sparse_mat = np.empty([num_rows + 1, num_columns + 1])
    sparse_mat[row, column] = values
    if user_or_item == 'item':
        sparse_mat = sparse_mat.T
    elif user_or_item == 'user':
        pass
    else:
        sys.exit()

    sparse_mat = csr_matrix(sparse_mat)
    return sparse_mat


if __name__ == '__main__':
    #ratings, user_counts, active_users = load_pandas()
    #detail_df = location_and_time_data()
    all_data_df = all_data()
