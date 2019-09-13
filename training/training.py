import pandas as pd
from resources import STRING
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.externals import joblib
import lightgbm as lgb

file_list = [filename for filename in os.listdir(STRING.train_processed) if
             filename.endswith('.csv')]

df = pd.read_csv(STRING.train_processed + file_list[0], sep=';', encoding='utf-8')

y = df[['event', 'crew']]
df = df.drop(['event', 'crew', 'experiment'], axis=1)

# Reescale
params_scale = []
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(df)
print(df.columns.values.tolist())
df_i = scaler.transform(df)
df = pd.DataFrame(scaler.transform(df), columns=df.columns)
print(df)


# Oversample?

# EDA
pca = PCA(whiten=True, svd_solver='randomized', n_components=len(df.columns.values.tolist()))
pca.fit(df_i)
cumsum = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
cumsum = list(cumsum)
print(cumsum)
var = [value for value in cumsum if value <= 99.99]
print(var)
pca_components = len(var)

pca = PCA(n_components=pca_components, whiten=True, svd_solver='randomized')
pca_components = pca.fit(df)
df = pd.DataFrame(pca_components.transform(df))
df = pd.concat([df, y['crew']], axis=1)

y = y.drop('crew', axis=1)
print(y['event'].value_counts())
parameters = {
    'boosting_type': 'goss',
    'max_leaves': 300,
    'max_depth': -1,
    'learning_rate': 0.01,
    'n_estimators': 300,
    'objective': 'multiclass',
    'class_weight': 'balanced',
    'random_state': 42

}

model = lgb.sklearn.LGBMClassifier()
model.set_params(**parameters)

model.fit(df, y)
dict_model = {'model': model, 'pca_components': pca_components, 'param_scale': scaler}

joblib.dump(dict_model,
            os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                         "pilot_fatalities", "models", "model.pkl"))