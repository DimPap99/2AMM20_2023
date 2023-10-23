import pandas as pd
import numpy as np
from causalnex.discretiser.discretiser_strategy import MDLPSupervisedDiscretiserMethod
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris["data"], iris["target"]
names = iris["feature_names"]
data = pd.DataFrame(X, columns=names)
data["target"] = y
discretiser = MDLPSupervisedDiscretiserMethod(
    {"min_depth": 0, "random_state": 2020, "min_split": 1e-3, "dtype": int}
)
discretiser.fit(
    feat_names=["sepal length (cm)"],
    dataframe=data,
    target="target",
    target_continuous=False,
)
discretised_data = discretiser.transform(data[["sepal length (cm)"]])
print(discretised_data.values.ravel())
