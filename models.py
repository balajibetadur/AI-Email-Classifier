'''
Author: Balaji Betadur
models.py :

Takes the parameters and return the model to the application.
Input: parameters (dictionary)
Output: model

1. add new model
2. add gridsearch parameters
3. fetch different hyper-parameters
4. All details about hyperparametsr is given in params.json
'''



# import packages
import pandas as pd
from sklearn.model_selection import GridSearchCV
import joblib
import json  
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as imbPipeline


# Defining Algorithm functions: Here the models are defined, any new model to the application can be added here.


def Logistic_Regression(parameters):    
       

        # fetching max_iter hyperparameter from the parameter dictionary
        max_iter = float(parameters['Maximum Iterations'])
        

        # model in a pipeline with count vectorizer and TfidfTransformer
        # model = Pipeline([('vect', CountVectorizer()),
        #         ('tfidf', TfidfTransformer()),
        #         ('model', LogisticRegression(n_jobs =  1, C = 1e5, max_iter = max_iter)),
        #        ])


        # model in a pipeline with count vectorizer and TfidfTransformer along with randmoversampler
        # Randomoversampler -> It balances the dataset by increasing the minority class samples randomly
        model = imbPipeline([
                ('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('oversample', RandomOverSampler()),
               ('clf', LogisticRegression(n_jobs=1, C=1e5, max_iter = max_iter)),
              ])


        # if gridsearch parameter is selected as yes then gridsearch cv code executes and tries
        # all given values for all given parameters
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
       

    #    fetching required hyperparameters from dictionary
        max_iter = float(parameters['Maximum Iteration'])
        loss = parameters['loss']
        alpha = float(parameters['alpha_svm'])
        penalty = parameters['penalty']
        

        # model in a pipeline with count vectorizer and TfidfTransformer
        # model = Pipeline([('vect', CountVectorizer()),
        #         ('tfidf', TfidfTransformer()),
        #         ('model', SGDClassifier(loss=loss, penalty=penalty,alpha=alpha, random_state=42, max_iter=max_iter, tol=None)),
        #        ])


        # model in a pipeline with count vectorizer and TfidfTransformer along with randmoversampler
        # Randomoversampler -> It balances the dataset by increasing the minority class samples randomly
        model = imbPipeline([
                ('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('oversample', RandomOverSampler()),
               ('clf', SGDClassifier(loss=loss, penalty=penalty,alpha=alpha, random_state=42, max_iter=max_iter, tol=None)),
                ])


        # if gridsearch parameter is selected as yes then gridsearch cv code executes and tries
        # all given values for all given parameters
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
       

       #    fetching required hyperparameters from dictionary
        alpha = float(parameters['alpha'])
        

        # model in a pipeline with count vectorizer and TfidfTransformer
        # model = Pipeline([('vect', CountVectorizer()),
        #        ('tfidf', TfidfTransformer()),
        #        ('model', MultinomialNB(alpha = alpha)),
        #       ])


        # model in a pipeline with count vectorizer and TfidfTransformer along with randmoversampler
        # Randomoversampler -> It balances the dataset by increasing the minority class samples randomly
        model = imbPipeline([
                ('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('oversample', RandomOverSampler()),
               ('clf', MultinomialNB(alpha = alpha)),
              ])


        # if gridsearch parameter is selected as yes then gridsearch cv code executes and tries
        # all given values for all given parameters
        if parameters["Grid Search"] == "Yes":
            params_ = {
                'model__alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
                }
            model = GridSearchCV(model, params_)

        return model

def Ranfom_Forest(parameters):    
        

        # model in a pipeline with count vectorizer and TfidfTransformer
        # model = Pipeline([('vect', CountVectorizer()),
        #        ('tfidf', TfidfTransformer()),
        #        ('model', RandomForestClassifier()),
        #       ])


        # model in a pipeline with count vectorizer and TfidfTransformer along with randmoversampler
        # Randomoversampler -> It balances the dataset by increasing the minority class samples randomly
        model = imbPipeline([
                ('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('oversample', RandomOverSampler()),
               ('clf', RandomForestClassifier()),
              ])


        # if gridsearch parameter is selected as yes then gridsearch cv code executes and tries
        # all given values for all given parameters
        if parameters["Grid Search"] == "Yes":
                
            params_ = {
                "ccp_alpha" : [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15],
                'max_depth': [80, 90, 100, 110],
                'min_samples_leaf': [1, 2],
                'min_samples_split': [2, 3],
                "criterion" : ["gini", "entropy"],
                # "max_features" : [int, float, "sqrt", "log", "auto"],
                'n_estimators': [100, 200, 300]       
            }
            model = GridSearchCV(model, params_)

        return model

