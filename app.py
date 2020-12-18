'''
Author: Balaji Betadur
app.py :

Flask app that recieve requests, process requests (traign ad predicting) and returns reponses
Input: data(training, predicting)
Output: results (trained model, predicted results)

1. upload data
2. preprocess data
3. train model
4. save model
5. predict test samples
6. generate results csv file
'''



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
            "4" : ["Linear Support Vector Machine",["1","2","8","9","10","11"]],   
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


# home function : first page to load when the website is opened
@app.route('/',methods=["GET","POST"])
def home():

    # initilalizing all required parameters 
    filenames = os.listdir(os.getcwd().replace('\\','/') + '/Models')

    mo = list(filenames)
    s = '$'.join(filenames)

    tabs=[]
    k = js = em = ' '
    l = len(tabs)
    if '' in mo:
        mo.remove('')

    for i in range(0,len(mo)):
        mo[i] = mo[i][:-4]

    subject = ""

    filenames.append('')
    filenames.append(' ')

    return render_template('index.html',subject = subject, js=js,em=em,l=l,params = all_params,mo=mo, s = s, filenames = filenames,k=k,tabs=tabs)


# prediction function: This function predicts the testing data
@app.route('/preee',methods=["GET","POST"])
def preee():
    if request.method=="POST":

        filenames = os.listdir(os.getcwd().replace('\\','/') + '/Models')
        ff = list(filenames)
        for i in ff:
            if i.endswith('-cv.pkl') or '.json' in i:
                filenames.remove(i)

        mo=list(filenames)
        s = '$'.join(filenames)
        js = request.form.get('semodel')
        email= request.form.get('title1')
        subject= request.form.get('subject')
        em=email 
        email2 = email + subject

        is_file = False
        try:  
            pat = request.files['fi2']
            pat.save(os.path.join(app.config['TESTS_FOLDER'],secure_filename( pat.filename)))
            is_file = True
        except:
            pass

        if is_file:

            test_csv = preprocess.process_test('TESTS',pat.filename)

        model_path = os.getcwd().replace('\\','/') + '/Models/' + js + '.pkl'
     
        model = joblib.load(model_path)


        if is_file:
            test_data = pd.read_excel(test_csv)
            # features = cv.fit_transform(test_data.Mail)
            result = model.predict(test_data.Mail)
            test_data['Predictions'] = result

            final_output = os.getcwd().replace('\\','/') + '/Results.csv'
            test_data.to_csv(final_output, index = None)
            results = test_data.values.tolist()
            
        else:
            # features = cv.fit_transform([email2])
            result = model.predict([email2])
            # print(classes)
            # print(result)
            results = [[email2 + '...', '',result[0]]]
            final_output = os.getcwd().replace('\\','/') + '/static/Results.xlsx'
            pd.DataFrame([[email2, result[0]]], columns = ['Email','Category']).to_csv(final_output, index = None)
            


        tabs = results
        # results = ''
        # tabs=[["Dear sir madam i duly completed the required pension application form ...","Retirement"]]
        l=len(tabs)
        k = 'OK'
        if em==None:
            em=''
        if '' in mo:
            mo = mo.remove('')
        for i in range(0,len(mo)):
            mo[i] = mo[i][:-4]
        return render_template('index.html',final_output = final_output, subject = subject,  results = results, k=k,l=l,params = all_params,mo=mo,s = s,em=em,js=js, filenames = filenames,tabs=tabs)

    return redirect("/")
    # return render_template('index.html',l = 0,)


# training function: This function trains model
@app.route('/sub',methods=["GET","POST"])
def sub():
    if request.method == "POST":
        json_ = eval(request.form.get('rs'))
        path = request.files.getlist('fi')
        t_time = json_['t_ime']
        data_files = []
        for i in path:
            data_files.append(i.filename)
            i.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(i.filename)))

        algorithm = all_params['models'][json_['t_ime']][json_['algo']][0]
        print(json_)
        print(algorithm)
        params = json_.copy()


        params['algo'] = algorithm
        params['files'] = data_files

        # split_ratio = round((params['Test Size'] / 100), 2)
        model_name = params['title']

        data_path = preprocess.process('Uploads', data_files)
        df = pd.read_excel(data_path)

        if t_time == '1':
            
            df['Mail'] = df['Mail'].apply(clean_text)

            x_train = df.Mail
            y_train = df.Class
            # x_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_ratio, random_state = 42)

            
            if algorithm == 'Logistic Regression':
                model = models.Logistic_Regression(params)
                print('trained')


            elif algorithm == 'Decesion Tree Classifier':
                pass

            elif algorithm == 'Random Forest Classifier':
                model = models.Random_Forest(params)
                print('trained')
                

            elif algorithm == 'Linear Support Vector Machine':
                model = models.SGD_Classifier(params)
                print('trained')
                

            elif algorithm == 'Naive Bayes Classifier':
                model = models.Multinomnal_NB(params)
                print('trained')

            model.fit(x_train, y_train)

            joblib.dump(model, os.getcwd().replace('\\','/') + f'/Models/{model_name}.pkl')
            print('model saved')

           
        elif t_time == '2': 

            if algorithm == 'Artificial Neural Networks':
                models.LogisticRegression(params)

            elif algorithm == 'LSTM':
                models.LogisticRegression(params)

            elif algorithm == 'BERT':
                models.LogisticRegression(params)

                
        
        
        # return 'hey'
    return redirect("/")


# Download function: This function downloads results
@app.route('/download',methods=["GET","POST"])
def download():
    return send_file('Results.csv',
                     mimetype='text/csv',
                     attachment_filename='Results.csv',
                     as_attachment=True)




if __name__ == "__main__":
    app.run(debug=True)
