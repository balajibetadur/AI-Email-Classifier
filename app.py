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
from bs4 import BeautifulSoup
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords 
import nltk
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

app = Flask(__name__)

all_params = {
    "time" : {
        "1": "short time",
        "2": "moderate time"
    },
    
    
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


    "models" : {

        "1" : {
            "1" : ["Logistic Regression",["1","2","4"]],
            "2" : ["Decesion Tree Classifier", ["1","2","3"]],
            "3" : ["Random Forest Classifier",["1","2","3","5","6"]],   
             # "4" : ["XGBoost",["1","2","3","4","6","7"]],   
            "4" : ["Linear Support Vector Machine",["1","2","8","9","10","11"]],   
            "5" : ["Naive Bayes Classifier",["1","2","7"]]
        },
        
        "2" : {
            "1" : ["Artificial Neural Networks",["7"]],
            "2" : ["LSTM",["7"]],
            "3" : ["BERT",["6","7"]]
        }
    }
}



TESTS_FOLDER = os.getcwd() + '\\TESTS'
UPLOAD_FOLDER = os.getcwd() + '\\Uploads'
ALLOWED_EXTENSIONS = {'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TESTS_FOLDER'] = TESTS_FOLDER
f = open ('params.json', "r") 



filenames = os.listdir(os.getcwd().replace('\\','/') + '/Models')
ff = list(filenames)
for i in ff:
    if i.endswith('-cv.pkl') or '.json' in i:
        filenames.remove(i)



# Reading from file 
params = json.loads(f.read()) 
dataframe = pd.read_excel('data_processed.xlsx')
filenames.append("")

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


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

@app.route('/',methods=["GET","POST"])
def home():

    filenames = os.listdir(os.getcwd().replace('\\','/') + '/Models')

    ff = list(filenames)
    for i in ff:
        if i.endswith('-cv.pkl') or '.json' in i:
            filenames.remove(i)

    # if request.method=="POST":
    mo=list(filenames)
    s = '$'.join(filenames)
    tabs=[]
    k='k'
    js='fjgjfhgjfhgjfhg'
    l=len(tabs)
    em=' '
    if '' in mo:
        mo.remove('')
    for i in range(0,len(mo)):
        mo[i] = mo[i][:-4]
    subject = ""
    filenames.append('')
    filenames.append(' ')
    return render_template('index.html',subject = subject, js=js,em=em,l=l,params = params,mo=mo, s = s, filenames = filenames,k=k,tabs=tabs)

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
            # ppp = os.getcwd().replace('\\','/')
            # for i in os.listdir(ppp + '/TESTS'):
            #     if os.path.isdir(ppp + '/TESTS/' + i):
            #         shutil.rmtree(ppp + '/TESTS/' + i) 
            #     else:
            #         os.remove(ppp + '/TESTS/' + i) 

            test_csv = preprocess.process_test('TESTS',pat.filename)

        model_path = os.getcwd().replace('\\','/') + '/Models/' + js + '.pkl'
        
        # vect = os.getcwd().replace('\\','/') + '/Models/' + js + '-cv.pkl'

        # model_info = open(os.getcwd().replace('\\','/') + '/Models/' + js + '.json')

        # model_data = json.load(model_info)

        # classes = model_data['Classes']

        # cv = CountVectorizer(decode_error="replace",vocabulary = pickle.load(open(vect, "rb")))

        model = joblib.load(model_path)

        # classes_ = os.list
        # dir(os.getcwd().replace('\\','/') + '/Uploads')
        # classes = []
        # for i in classes_:
        #     if not i.endswith('.zip'):
        #         classes.append(i)

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
        return render_template('index.html',final_output = final_output, subject = subject,  results = results, k=k,l=l,params = params,mo=mo,s = s,em=em,js=js, filenames = filenames,tabs=tabs)

    return redirect("/")
    # return render_template('index.html',l = 0,)

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

            # elif algorithm == 'Random Forest Classifier':
            #     model = models.Random_Forest(params)
            #     print('trained')
                

            elif algorithm == 'Linear Support Vector Machine':
                model = models.SGD_Classifier(params)
                print('trained')
                

            elif algorithm == 'Naive Bayes Classifier':
                model = models.Multinomnal_NB(params)
                print('trained')

            model.fit(x_train, y_train)

            joblib.dump(model, os.getcwd().replace('\\','/') + f'/Models/{model_name}.pkl')
            print('model saved')
            # paras = model.get_params()
            # try:
            #     # paras.pop('estimator')
            #     paras.pop('steps')
            #     paras.pop('model')
            #     paras.pop('tfidf')
            #     paras.pop('vect')
            # except Exception as e:
            #     print(e)
                

            # dictionary ={   
            # # "Accuracy" : max(score_1, score_2),
            # "Model" : model_name,
            # "Classes" : df['Class'].unique().tolist(),
            # "Parameters" : paras
            # }   

            # # pickle.dump(cv.vocabulary_,open( os.getcwd().replace('\\','/') + f"/Models/{model_name}-cv.pkl","wb"))  
            
            # with open( os.getcwd().replace('\\','/') + f"/Models/{model_name}.json", "w") as outfile:  
            #     json.dump(dictionary, outfile) 
            # print('json saved')
            
        elif t_time == '2': 

            if algorithm == 'Artificial Neural Networks':
                models.LogisticRegression(params)

            elif algorithm == 'LSTM':
                models.LogisticRegression(params)

            elif algorithm == 'BERT':
                models.LogisticRegression(params)

                
        
        
        # return 'hey'
    return redirect("/")

@app.route('/download',methods=["GET","POST"])
def download():
    return send_file('Results.csv',
                     mimetype='text/csv',
                     attachment_filename='Results.csv',
                     as_attachment=True)




if __name__ == "__main__":
    app.run(debug=True)
