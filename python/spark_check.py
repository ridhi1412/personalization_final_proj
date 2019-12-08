# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:08:05 2019

@author: aksmi
"""

#from pyspark.sql.types import *
#from pyspark.sql.types import StructField, StructType
#from pyspark import SparkContext
#from pyspark.ml.evaluation import RegressionEvaluator
#from pyspark.ml.recommendation import ALS
#from pyspark.sql.functions import explode

from pyspark.rdd import RDD, _prepare_for_python_RDD, ignore_unicode_prefix
from pyspark.serializers import AutoBatchedSerializer, PickleSerializer
#from pyspark.sql import since
from pyspark.sql.types import Row, StringType, StructType
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.readwriter import DataFrameReader
from pyspark.sql.utils import install_exception_handler
from pyspark.sql.functions import UserDefinedFunction

try:
    import pandas
    has_pandas = True
except Exception:
    has_pandas = False

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

l = [('Alice', 1)]
sqlContext.createDataFrame(l).collect()
[Row(_1=u'Alice', _2=1)]
sqlContext.createDataFrame(l, ['name', 'age']).collect()
[Row(name=u'Alice', age=1)]

d = [{'name': 'Alice', 'age': 1}]
sqlContext.createDataFrame(d).collect()
[Row(age=1, name=u'Alice')]

rdd = sc.parallelize(l)
sqlContext.createDataFrame(rdd).collect()
[Row(_1=u'Alice', _2=1)]
df = sqlContext.createDataFrame(rdd, ['name', 'age'])
df.collect()
[Row(name=u'Alice', age=1)]

from pyspark.sql import Row
Person = Row('name', 'age')
person = rdd.map(lambda r: Person(*r))
df2 = sqlContext.createDataFrame(person)
df2.collect()
[Row(name=u'Alice', age=1)]

from pyspark.sql.types import *
schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True)
])
df3 = sqlContext.createDataFrame(rdd, schema)
df3.collect()
