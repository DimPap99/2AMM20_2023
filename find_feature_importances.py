import pandas as pd
import os
from helpers import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
import numpy as np
from joblib import parallel_backend

DATASET_BASE = "dataset"
PICKLED_OBJS = "pickled_objs"

txs_classes:pd.DataFrame = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_classes.csv"), ret_Dataframe=True)
txs_edgelist:pd.DataFrame = read_csv(os.path.join(DATASET_BASE, "elliptic_txs_edgelist.csv"), ret_Dataframe=True)
txs_features:pd.DataFrame =  read_csv(os.path.join(DATASET_BASE, "elliptic_txs_features.csv"), ret_Dataframe=True)


merge_feats_class = pd.merge(txs_features, txs_classes, on='txId').dropna()

merge_feats_class = merge_feats_class.loc[merge_feats_class['class'].isin([1, 2])].reset_index(drop=True)


merge_feats_class = merge_feats_class.drop('txId',axis=1)
merge_feats_class = merge_feats_class.drop('Time step',axis=1)

X = merge_feats_class.drop('class', axis=1)
y = merge_feats_class['class']

# Split the data into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize a Random Forest classifier (you can use your preferred classifier)
clf = RandomForestClassifier(n_estimators=50, random_state=42)

# Initialize RFECV with cross-validation (change cv value as needed)
rfecv = RFECV(estimator=clf, step=1, cv=2, scoring='accuracy', verbose=1)

with parallel_backend("threading"):
# Fit RFECV to your training data
    rfecv.fit(X_train, y_train)

# Get the selected feature indices
selected_feature_indices = rfecv.support_

feature_names = list(merge_feats_class.columns[np.where(selected_feature_indices)].values)
importances = list(rfecv.estimator_.feature_importances_)
print(len(feature_names))

feature_df = pd.DataFrame([importances], columns=feature_names)
from helpers import pickle_data
pickle_data("latest_feature_importances.pkl", feature_df.iloc[0].to_dict())
#print(feature_df)



