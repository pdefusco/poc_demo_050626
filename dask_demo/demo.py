#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
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
# #  Author(s): Paul de Fusco
#***************************************************************************/

import dask.dataframe as dd
import numpy as np
import pandas as pd

from dask_ml.datasets import make_classification
from dask_ml.linear_model import LogisticRegression
from dask_ml.ensemble import RandomForestClassifier

# Create dataset
X, y = make_classification(n_samples=5000, n_features=10, chunks=5)

# Convert to Dask DataFrame
df = X.to_dask_dataframe()
df["label"] = y

# Train models
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100)

lr.fit(X, y)
rf.fit(X, y)

# Predictions
df["score_lr"] = lr.predict_proba(X)[:, 1]
df["score_rf"] = rf.predict_proba(X)[:, 1]

pdf = df.compute()  # bring to pandas for plotting

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
    lift_df["lift"] = lift_df["event_rate"] / df["y"].mean()

    return lift_df.reset_index()

lift_lr = lift_table(pdf["label"], pdf["score_lr"])
lift_rf = lift_table(pdf["label"], pdf["score_rf"])

import matplotlib.pyplot as plt

plt.plot(lift_lr["bin"], lift_lr["lift"], marker="o", label="LR")
plt.plot(lift_rf["bin"], lift_rf["lift"], marker="o", label="RF")

plt.axhline(1, linestyle="--")
plt.title("Double Lift Chart (Dask)")
plt.legend()
plt.show()

def lorenz_curve(y, score):
    df = pd.DataFrame({"y": y, "score": score})
    df = df.sort_values("score")

    df["cum_y"] = df["y"].cumsum()
    df["cum_pop"] = np.arange(1, len(df)+1)

    lx = df["cum_pop"] / len(df)
    ly = df["cum_y"] / df["y"].sum()

    return lx, ly

def gini(lx, ly):
    return 1 - 2 * np.trapz(ly, lx)

lx_lr, ly_lr = lorenz_curve(pdf["label"], pdf["score_lr"])
lx_rf, ly_rf = lorenz_curve(pdf["label"], pdf["score_rf"])

print("Gini LR:", gini(lx_lr, ly_lr))
print("Gini RF:", gini(lx_rf, ly_rf))
