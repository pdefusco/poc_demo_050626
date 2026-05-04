from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import cml.data_v1 as cmldata

# Sample in-code customization of spark configurations
#from pyspark import SparkContext
#SparkContext.setSystemProperty('spark.executor.cores', '1')
#SparkContext.setSystemProperty('spark.executor.memory', '2g')

CONNECTION_NAME = "cf-aw-dl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# Sample usage to run query through spark
EXAMPLE_SQL_QUERY = "show databases"
spark.sql(EXAMPLE_SQL_QUERY).show()

df = spark.read.csv("data/preprocessed_flight_data.csv", header=True, inferSchema=True)

df = df.dropna()

feature_cols = ["uniquecarrier", "origin", "dest", "week", "hour"]
target_col = "cancelled"

df = df.select(*feature_cols, target_col)

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

categorical_cols = ["uniquecarrier", "origin", "dest"]
numeric_cols = ["week", "hour"]

# Step 1: StringIndex
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_cols
]

# Step 2: OneHotEncode
encoders = [
    OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
    for c in categorical_cols
]

assembler = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in categorical_cols] + numeric_cols,
    outputCol="features"
)

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

lr = LogisticRegression(
    featuresCol="features",
    labelCol=target_col
)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol=target_col,
    numTrees=100
)

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=indexers + encoders + [assembler])

train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

pipeline_model = pipeline.fit(train_df)

train_prepared = pipeline_model.transform(train_df)
test_prepared = pipeline_model.transform(test_df)

pipeline_model = pipeline.fit(train_df)

train_prepared = pipeline_model.transform(train_df)
test_prepared = pipeline_model.transform(test_df)

lr_model = lr.fit(train_prepared)
rf_model = rf.fit(train_prepared)

pred_lr = lr_model.transform(test_prepared)
pred_rf = rf_model.transform(test_prepared)

from pyspark.sql.functions import col

pred_lr = pred_lr.withColumn("score_lr", col("probability")[1])
pred_rf = pred_rf.withColumn("score_rf", col("probability")[1])

# Convert to pandas for plotting
pdf = pred_lr.select("label", "score_lr") \
    .join(pred_rf.select("score_rf"), on=None) \
    .toPandas()

def lift_table(y_true, y_score, bins=10):
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    df["bin"] = pd.qcut(df.index, bins, labels=False)

    grouped = df.groupby("bin")
    lift_df = pd.DataFrame({
        "events": grouped["y"].sum(),
        "total": grouped["y"].count()
    })

    lift_df["event_rate"] = lift_df["events"] / lift_df["total"]
    overall_rate = df["y"].mean()
    lift_df["lift"] = lift_df["event_rate"] / overall_rate

    return lift_df.reset_index()

lift_lr = lift_table(pdf["label"], pdf["score_lr"])
lift_rf = lift_table(pdf["label"], pdf["score_rf"])

plt.figure(figsize=(8,5))

plt.plot(lift_lr["bin"], lift_lr["lift"], marker="o", label="Logistic Regression")
plt.plot(lift_rf["bin"], lift_rf["lift"], marker="o", label="Random Forest")

plt.axhline(1, linestyle="--", color="gray")
plt.xlabel("Decile (0 = highest score)")
plt.ylabel("Lift")
plt.title("Double Lift Chart (PySpark Models)")
plt.legend()
plt.show()

def lorenz_curve(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df = df.sort_values("score")

    df["cum_y"] = df["y"].cumsum()
    df["cum_pop"] = np.arange(1, len(df)+1)

    total_y = df["y"].sum()
    total_pop = len(df)

    lx = df["cum_pop"] / total_pop
    ly = df["cum_y"] / total_y

    return lx, ly

def gini_from_lorenz(lx, ly):
    area = np.trapz(ly, lx)
    return 1 - 2 * area

lx_lr, ly_lr = lorenz_curve(pdf["label"], pdf["score_lr"])
lx_rf, ly_rf = lorenz_curve(pdf["label"], pdf["score_rf"])

gini_lr = gini_from_lorenz(lx_lr, ly_lr)
gini_rf = gini_from_lorenz(lx_rf, ly_rf)

print("Gini LR:", gini_lr)
print("Gini RF:", gini_rf)

plt.figure(figsize=(6,6))

plt.plot(lx_lr, ly_lr, label=f"LR (Gini={gini_lr:.3f})")
plt.plot(lx_rf, ly_rf, label=f"RF (Gini={gini_rf:.3f})")

plt.plot([0,1], [0,1], linestyle="--", color="gray")

plt.xlabel("Cumulative Population")
plt.ylabel("Cumulative Events")
plt.title("Lorenz Curve (PySpark Models)")
plt.legend()
plt.show()
