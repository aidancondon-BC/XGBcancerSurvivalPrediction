import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('METABRIC_RNA_mutation.csv', engine='python')

features = list(df.columns)
features.remove('overall_survival_months')
features.remove('overall_survival')
features.remove('death_from_cancer')

X = df[features]
y = df.overall_survival

s = (X.dtypes == 'object')
object_cols = list(s[s].index)
label_X = X.copy()
label_X[object_cols] = OrdinalEncoder().fit_transform(X[object_cols])

X = label_X
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

fitted_model = XGBClassifier(n_estimators=33, learning_rate=0.1)
fitted_model.fit(train_X, train_y)
final_pred = fitted_model.predict(val_X)
print(f'ACC: {accuracy_score(val_y, final_pred)}')
