import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import texthero as hero
import pickle
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer,HashingVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score
from datetime import datetime

from spacy.lang.es.stop_words import STOP_WORDS
spacy_stopwords = list(STOP_WORDS)

print("LOADING DATA : ",datetime.now())
data = pd.read_csv("data/domain_clf_data_2.csv").dropna()
print("END          : ",datetime.now())

X = data["title_clean"]
y = data["domain_id"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


tokens = '[A-Za-z0-9]+(?==\\s+)'
lr_1vsall_hash_text_clf =  Pipeline([('Vectorizer', HashingVectorizer(
                                                               )),
                                ('clf',  OneVsRestClassifier(LogisticRegression()))])

print("HashOneVsRestClassifier_LogisticRegression-->START TRAINING AT : ",datetime.now())
lr_clf=lr_1vsall_hash_text_clf.fit(X_train,y_train)
print("HashOneVsRestClassifier_LogisticRegression-->END   TRAINING AT : ",datetime.now())
y_pred = lr_clf.predict(X_test)
print("balanced_accuracy_score : ",balanced_accuracy_score(y_test, y_pred))


# print("HashOneVsRestClassifier_LogisticRegression-->balanced_accuracy_score : ",balanced_accuracy_score(y_test, y_pred))
# dump(lr_clf, 'models/HashOneVsRestClassifier_LogisticRegression.joblib')
#
# SGDClassifier_text_clf =  Pipeline([('TfidfVectorizer', TfidfVectorizer(encoding='latin-1',
#                                                                max_df=0.90,
#                                                                min_df=5,
#                                                                max_features=100000,
#                                                                stop_words=list(spacy_stopwords),
#                                                               )),
#                                 ('SGDClassifier',  SGDClassifier(class_weight="balanced"))])
#
# print("SGDClassifier-->START TRAINING AT : ",datetime.now())
# lr_clf=SGDClassifier_text_clf.fit(X_train,y_train)
# print("SGDClassifier-->END   TRAINING AT : ",datetime.now())
# y_pred = lr_clf.predict(X_test)
#
# print("SGDClassifier-->balanced_accuracy_score : ",balanced_accuracy_score(y_test, y_pred))
# dump(lr_clf, 'models/SGDClassifier.joblib')
#
#
# lr_1vsall_text_clf =  Pipeline([('TfidfVectorizer', TfidfVectorizer(encoding='latin-1',
#                                                                max_df=0.90,
#                                                                min_df=5,
#                                                                max_features=100000,
#                                                                stop_words=list(spacy_stopwords),
#                                                               )),
#                         ('clf',  OneVsRestClassifier(LogisticRegression()))])
#
# print("OneVsRestClassifier_LogisticRegression-->START TRAINING AT : ",datetime.now())
# lr_clf=lr_1vsall_text_clf.fit(X_train,y_train)
# print("OneVsRestClassifier_LogisticRegression-->END   TRAINING AT : ",datetime.now())
# y_pred = lr_clf.predict(X_test)
#
# print("OneVsRestClassifier_LogisticRegression-->balanced_accuracy_score : ",balanced_accuracy_score(y_test, y_pred))
# dump(lr_clf, 'models/OneVsRestClassifier_LogisticRegression.joblib')
#
#
# sgd_1vsall_hash_text_clf =  Pipeline([('Vectorizer', HashingVectorizer(token_pattern=tokens,
#                                                                norm=None,
#                                                                binary=False,
#                                                                alternate_sign=False
#                                                                )),
#                                 ('clf',  OneVsRestClassifier(SGDClassifier(class_weight="balanced")))])
#
# print("HashOneVsRestClassifier_SGDClassifier-->START TRAINING AT : ",datetime.now())
# lr_clf=sgd_1vsall_hash_text_clf.fit(X_train,y_train)
# print("HashOneVsRestClassifier_SGDClassifier-->END   TRAINING AT : ",datetime.now())
# y_pred = lr_clf.predict(X_test)
#
# print("HashOneVsRestClassifier_SGDClassifier-->balanced_accuracy_score : ",balanced_accuracy_score(y_test, y_pred))
# dump(lr_clf, 'models/HashOneVsRestClassifier_SGDClassifier.joblib')
