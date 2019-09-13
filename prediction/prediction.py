from sklearn.externals import joblib
import os
from resources import STRING
import pandas as pd

dict_models = joblib.load(
        os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     "pilot_fatalities", "models", "model.pkl"))

model = dict_models.get("model")
pca_comp = dict_models.get("pca_components")
scale_param = dict_models.get("param_scale")

df = pd.read_csv(STRING.test, sep=',', encoding='utf-8')
print(df)
df = df.drop('experiment', axis=1)

y = df[['crew', 'id']]
df = df.drop(['crew', 'id'], axis=1)
print(df.columns.values.tolist())
# Scale
df = pd.DataFrame(scale_param.transform(df), columns=df.columns)
print(df)

df = pca_comp.transform(df)
df = pd.DataFrame(df)
df = pd.concat([df, y['crew']], axis=1)
prediction = model.predict(df)
prediction = pd.DataFrame(prediction)
prediction = pd.get_dummies(prediction)
print(prediction)
prediction.columns = ['A', 'B', 'C', 'D']
df = pd.concat([y[['id']], prediction], axis=1)
print(prediction)
df.to_csv(STRING.submission, index=False, sep=',', encoding='utf-8')
# falta atar ID


