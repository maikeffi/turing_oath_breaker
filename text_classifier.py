#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 18:15:05 2018

@author: root
"""
import re
import pandas as pd
import time
from collections import Counter
import numpy as np
import nltk 
nltk.data.path.append("/prod/nltk_data/")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from textblob import Word
from pywsd.utils import lemmatize_sentence
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer ,TfidfTransformer
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals.joblib import Parallel, delayed


start_time = time.time()
stop_words = set(stopwords.words('english'))
stop_words.add('The')
stop_words.add('This')
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

data = pd.read_csv('/prod/starlord/turing_oath_breaker/data/ads_en_us.csv')

def remove_stop_words(sentence):
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    sentence = ' '.join(filtered_sentence)
    return sentence

def perform_lemm(sentence):
    word_tokens = word_tokenize(sentence)
    lemm_sentence = [lemmatizer.lemmatize(w,pos='v') for w in word_tokens]
    return ' '.join(lemm_sentence)

def clean_str(s):
    """Clean sentence"""
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
    s = re.sub(r'[^\x00-\x7F]+', "", s)
    s = re.sub(r'[^\w\s]',"",s)
    return s.strip().lower()

"""
Remove Duplicates
"""
data = data.drop_duplicates(subset='value',keep='last')
"""
Remove Skew
"""
data_27 = data.loc[data.subcatid == 27, :]
data_27s = data_27.sample(frac=0.5)
data1 = data[data.subcatid != 27]
data_s = pd.concat([data1, data_27s], axis = 0 )
data_bal=data_s.groupby('subcatid').filter(lambda x: len(x) >= 100)
"""
#data_bal['value_processed'] = data_s.value.apply(clean_str)
#data_bal['value_processed_lemm'] = data_bal.value_processed.apply(perform_lemm)
"""
data_bal['value_processed'] = Parallel(n_jobs=-1, verbose=10)(delayed(clean_str)(i) for i in data_bal['value'].tolist())
data_bal['value_processed_lemm'] = Parallel(n_jobs=-1, verbose=10)(delayed(perform_lemm)(i) for i in data_bal['value_processed'].tolist())

print(data_bal.value_processed_lemm.head(20))


X = data_bal.value_processed
y = data_bal.subcatid
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 5)

"""
# Number of trees in random forest
"""
n_estimators = [200,300]
"""
# Maximum number of levels in tree
"""
max_depth =[150,200]
"""
#max_depth.append(None)
# Minimum number of samples required to split a node
"""
min_samples_split = [10,15]
"""
# Minimum number of samples required at each leaf node
"""
min_samples_leaf = [1]

"""
# Create the random grid
"""
random_grid = {
               'clf__n_estimators': n_estimators,
               'clf__max_depth': max_depth,
               'clf__min_samples_split': min_samples_split,
               'clf__min_samples_leaf': min_samples_leaf
              }

#preparing the final pipeline using the selected parameters
model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', RandomForestClassifier(max_features='auto',bootstrap=False,random_state = 42))])

gs_clf_rfc = GridSearchCV(model, random_grid,cv=2,verbose=10, n_jobs=-1)
gs_clf_rfc.fit(X_train, y_train)

model = gs_clf_rfc.best_estimator_
y_pred = model.predict(X_test)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print("--- %s seconds --- " %(time.time()-start_time))
