from sklearn.model_selection import train_test_split
from resources.challenge_metric import ndcg_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from joblib import dump, load

import pandas as pd

from datetime import datetime


data = pd.read_csv("data/domain_predict.csv")

numeric_features = ['len_user_history', 'number_search',"number_view"]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['max_view_domain_id', 'most_domain_view', 'last_view_domain_id','last_search_domain']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

clf_domain = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])


X = data.drop("domain_id",axis=1)
y = data["domain_id"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

print("START TRAINING : ", datetime.now())

clf_domain.fit(X_train, y_train)

print("END  TRAINING : ", datetime.now())


print("model score: %.3f" % clf_domain.score(X_test, y_test))


dump(clf_domain, 'models/Domain_pred_LogisticRegression.joblib')
