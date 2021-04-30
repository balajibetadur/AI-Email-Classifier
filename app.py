###################################################################################################################################
#Author: Balaji Betadur and prasanna kusugal
#app.py :
#
#Flask app that recieve requests, process requests (traign ad predicting) and returns reponses
#Input: data(training, predicting)
#Output: results (trained model, predicted results)
#
#1. upload data
#2. preprocess data
#3. train model
#. save model
#5. predict test samples
#6. generate results csv file
###################################################################################################################################



# import packages
from flask import Flask,render_template,request,redirect,send_file
import pandas as pd
import models
import joblib
import pickle
import preprocess
import shutil
import os
import time
import numpy as np
from numpy import random
from werkzeug.utils import secure_filename
import json
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
import json
import re
from bs4 import BeautifulSoup


# create flask app
app = Flask(__name__)


# all parameters including filter params, algorithms, and parameters for all algoritms
all_params = {

    # time si the filter criteria for algorithms
    "time" : {
        "1": "short time",
        "2": "moderate time"
    },
    
    
    # all the parameters along with their default values, possible values, and small explanation about the parameter
    "parameters" : {
        "1" : ["Test Size","input","20","Enter test size. ex: 20 for 20% test set"],
        "2" : ["Grid Search","select",["No","Yes"],"Tests for different hyperparameter values (Increases training time)"],
        "3" : ["Number of Estimators", "input","100","(Number fo Trees) Suggested value is between 64 - 128 trees. Huge value may increase training time"],
        "4" : ["Maximum Iterations","input","100","Maximum number of iterations taken for the solvers to converge"],
        "5" : ["Min Samples Split","input","2","The minimum number of samples required to split an internal node"],
        "6" : ["Min Samples Leaf", "input", "1", "The minimum number of samples required to be at a leaf node."],
        "7" : ["alpha", "select",["1","0", "0.1", "0.01", "0.001", "0.0001", "0.00001"], "Additive (Laplace/Lidstone) smoothing parameter(0 for no smoothing)"],
        "8" : ["loss", "select", ["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "epsilon_insensitive"], "The loss function to be used" ],
        "9" : ["penalty", "select", ["l1","l2","elasticnet"], "The penalty (aka regularization term) to be used"],
        "10": ["alpha_svm", "select",["0.001","0.01","0.0001"], "Constant that multiplies the regularization term."],
        "11": ["Maximum Iteration","input","5", "The maximum number of passes over the training data (aka epochs)"]
    },


    # models list along with the parameters required
    "models" : {
        
        # for short time criteria
        "1" : {
            "1" : ["Logistic Regression",["1","2","4"]],
            "2" : ["Decesion Tree Classifier", ["1","2","3"]],
            "3" : ["Random Forest Classifier",["1","2","3","5","6"]],   
            "4" : ["Support Vector Machine",["1","2","8","9","10","11"]],   
            "5" : ["Naive Bayes Classifier",["1","2","7"]]
        },
        
        # for moedrate time criteria
        "2" : {
            "1" : ["Artificial Neural Networks",["7"]],
            "2" : ["LSTM",["7"]],
            "3" : ["BERT",["6","7"]]
        }
    }
}


# test folder is where all the test data is saved
TESTS_FOLDER = os.getcwd() + '\\TESTS'
app.config['TESTS_FOLDER'] = TESTS_FOLDER

# uploads folder is where all the train data is saved
UPLOAD_FOLDER = os.getcwd() + '\\Uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# allowed extensions allow you to see only zip files whie uploading files
ALLOWED_EXTENSIONS = {'zip'}



# filenames is a list of all models to make sure the name given for new model is unique
filenames = os.listdir(os.getcwd().replace('\\','/') + '/Models')


# Reading training preprocesssed data file 
dataframe = pd.read_excel('data_processed.xlsx')
filenames.append("")

# all punctuations to remove
REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


# function to clean text
# input: raw text
# output: cleaned text
def clean_text(text):
                """
                    text: a string
                    
                    return: modified initial string
                """
                text = BeautifulSoup(text, "lxml").text # HTML decoding
                text = text.lower() # lowercase text
                text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
                text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
                text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
                return text


# training function: This function trains model
@app.route('/sub',methods=["GET","POST"])
def sub():
    if request.method == "POST":

        # get parameters from front end
        json_ = eval(request.form.get('rs'))
        paths = request.files.getlist('fi')
        t_time = json_['t_ime']

        # list to save all file paths for training data
        data_files = []
        for path in paths:

            # adding the path in a list
            data_files.append(i.filename)

            # saving the path
            path.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(path.filename)))

        # fethcing the selected algorithm from parameters
        algorithm = all_params['models'][json_['t_ime']][json_['algo']][0]

        # copying the json to a new variable
        params = json_.copy()

        # getting the algorithm
        params['algo'] = algorithm

        # getting the paths
        params['files'] = data_files

        # split_ratio = round((params['Test Size'] / 100), 2)
        model_name = params['title']

        # calling preprocess function to process training data
        data_path = preprocess.process('Uploads', data_files)

        # reading the processed data
        df = pd.read_excel(data_path)

        # if training time is short
        if t_time == '1':
            
            # get mail from csv
            df['Mail'] = df['Mail'].apply(clean_text)

            # split training and testing data
            x_train = df.Mail
            y_train = df.Class
            # x_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, random_state = 42)

            # if selected algorithm is logistic Regression
            if algorithm == 'Logistic Regression':

                # call logistic regression and save the model in the variable
                model = models.Logistic_Regression(params)


            # if selected algorithm is logistic Regression
            elif algorithm == 'Decesion Tree Classifier':
                
                # call logistic regression and save the model in the variable
                model = models.Decesion_Tree(params)
                

            # if selected algorithm is logistic Regression
            elif algorithm == 'Random Forest Classifier':
                
                # call logistic regression and save the model in the variable
                model = models.Random_Forest(params)
                

            # if selected algorithm is logistic Regression
            elif algorithm == 'Support Vector Machine':
                
                # call logistic regression and save the model in the variable
                model = models.SGD_Classifier(params)
                

            # if selected algorithm is logistic Regression
            elif algorithm == 'Naive Bayes Classifier':
                
                # call logistic regression and save the model in the variable
                model = models.Multinomnal_NB(params)

            # training the model
            model.fit(x_train, y_train)

            # saving the model in models folder
            joblib.dump(model, os.getcwd().replace('\\','/') + f'/Models/{model_name}.pkl')

       
        # if training time is long
        elif t_time == '2': 

            # if selected algorithm is logistic Regression
            # can include deep learning models here
            pass

                
        
        
        # return 'hey'
    return redirect("/")


# Download function: This function downloads results
@app.route('/download',methods=["GET","POST"])
def download():
    # to download the results after prediction
    return send_file('Results.csv',
                     mimetype='text/csv',
                     attachment_filename='Results.csv',
                     as_attachment=True)




if __name__ == "__main__":
    app.run(debug=True)
