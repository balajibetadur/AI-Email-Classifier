import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import json  
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
# import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as imbPipeline

def Logistic_Regression(parameters):    
       
        max_iter = float(parameters['Maximum Iterations'])
        
        # model = Pipeline([('vect', CountVectorizer()),
        #         ('tfidf', TfidfTransformer()),
        #         ('model', LogisticRegression(n_jobs =  1, C = 1e5, max_iter = max_iter)),
        #        ])

        model = imbPipeline([
                ('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('oversample', RandomOverSampler()),
               ('clf', LogisticRegression(n_jobs=1, C=1e5, max_iter = max_iter)),
              ])

        if parameters["Grid Search"] == "Yes":
            params_ = {
                # 'dual' : [True, False],
                'model__tol': [1e-4, 1e-5, 5e-4],
                'model__fit_intercept':[True, False],
                'model__class_weight': [None, 'balanced'],
                'model__solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'model__max_iter' : [100, 150, 500]
                }
            model = GridSearchCV(model, params_)

        return model

def SGD_Classifier(parameters):    
       
        max_iter = float(parameters['Maximum Iteration'])
        loss = parameters['loss']
        alpha = float(parameters['alpha_svm'])
        penalty = parameters['penalty']
        
        model = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('model', SGDClassifier(loss=loss, penalty=penalty,alpha=alpha, random_state=42, max_iter=max_iter, tol=None)),
               ])

        if parameters["Grid Search"] == "Yes":
            params_ = {                
                'model__tol': [1e-3, 1e-2, 5e-4],
                'model__loss':["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "epsilon_insensitive"],
                'model__penalty': ["l1","l2","elasticnet"],
                'model__alpha' : [0.01, 0.001, 0.0001],
                'model__max_iter' : [5, 100, 500, 1000]
                }
            model = GridSearchCV(model, params_)

        return model
       
def Multinomnal_NB(parameters):    
       
        alpha = float(parameters['alpha'])
        
        # model = Pipeline([('vect', CountVectorizer()),
        #        ('tfidf', TfidfTransformer()),
        #        ('model', MultinomialNB(alpha = alpha)),
        #       ])

        model = imbPipeline([
                ('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('oversample', RandomOverSampler()),
               ('clf', MultinomialNB(alpha = alpha)),
              ])

        if parameters["Grid Search"] == "Yes":
            params_ = {
                'model__alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
                }
            model = GridSearchCV(model, params_)

        return model

