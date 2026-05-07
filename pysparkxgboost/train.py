#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Jonathan Ingalls, Paul de Fusco
#***************************************************************************/

#!pip install onnx onnxmltools skl2onnx
from pyspark import SparkContext
from pyspark.sql import SparkSession
from xgboost.spark import SparkXGBClassifier
import cml.data_v1 as cmldata
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import LongType, FloatType, IntegerType, StringType, \
                              DoubleType, BooleanType, ShortType, \
                              TimestampType, DateType, DecimalType, \
                              ByteType, BinaryType, ArrayType, MapType, \
                              StructType, StructField

# Sample in-code customization of spark configurations
SparkContext.setSystemProperty('spark.executor.cores', '2')
SparkContext.setSystemProperty('spark.executor.memory', '4g')
SparkContext.setSystemProperty('spark.executor.instances', '4')
SparkContext.setSystemProperty('spark.driver.cores', '2')
SparkContext.setSystemProperty('spark.driver.memory', '4g')
SparkContext.setSystemProperty('spark.dynamicAllocation.enabled', 'false')

CONNECTION_NAME = "cf-aw-dl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

data = [
    (0.0, 1.0, 0),
    (1.0, 2.0, 0),
    (2.0, 1.0, 1),
    (3.0, 3.0, 1),
]

df = spark.createDataFrame(data, ["x1", "x2", "label"])

assembler = VectorAssembler(
    inputCols=["x1", "x2"],
    outputCol="features"
)

train_df = assembler.transform(df).select("features", "label")

xgb = SparkXGBClassifier(
    features_col="features",
    label_col="label",
    num_workers=2,
    max_depth=3,
    n_estimators=10,
)

model = xgb.fit(train_df)

import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Save booster temporarily
booster.save_model("/home/cdsw/xgb.json")

# Re-load into standard XGBoost classifier
native_model = xgb.XGBClassifier()
native_model.load_model("/home/cdsw/xgb.json")

initial_type = [
    ("features", FloatTensorType([None, 2]))
]

onnx_model = convert_xgboost(
    native_model,
    initial_types=initial_type
)

with open("/home/cdsw/xgb_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
