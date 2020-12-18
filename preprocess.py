'''
Author: Balaji Betadur
preprocess.py :

Takes data, unzips it, xtract text from .msg files, process it, formats it and converts it to a csv
Input: path (zip file)
Output: csv path

1. unzip file
2. extract text from ".msg" files
3. cleans data
4. formats data
5. generates csv for a formatted data
'''

# importing packages
import zipfile
import pandas as pd
import extract_msg
import re
import os
import random
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 


# function to unzip the zipped file
# input: path of zip file
# output: unzipped folder path
def unzip(zip_path):
    archive = zipfile.ZipFile(zip_path, 'r')
    
    dest = zip_path.split('/')[:-1]
    dest = '/'.join(dest)
    for file in archive.namelist():
        archive.extract(file, dest)     


# function to clean text includes stopwords removal, stemming, punctuation removal
# input: raw text
# output: cleaned text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.
        # Convert words to lower case and split them
        text = str(text).lower().split()
        # Optionally, remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
        text = " ".join(text)
        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)

        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        # Optionally, shorten words to their stems
        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)
        # Return a list of words
        return(text)


# function to extract text from ".msg" files
# input: path of .msg file
# output: list of mail subject, mail body, classname and filename
def extract(path):
    msg = extract_msg.Message(path)
    n = (msg.subject + msg.body).index('Subject:') + 9
    return [text_to_wordlist((msg.subject + msg.body)[n:].replace('\n',' ').replace('\r',' ')) , path.split('/')[-2], path.split('/')[-1]]
    # return [msg.subject,msg.body[n:].replace('\n',' ').replace('\r',' '), path.split('/')[-2]]


# function to process training data and generate the CSV file in required format
# input: filenames and base path
# output: csv
def process(base, filenames):
        
    data = []

    # loop to unzip all folders and extract text from all .msg files
    for filename in filenames:
        path = os.getcwd().replace('\\','/') + '/' + base 
        
        unzip(path + '/' + filename)
        
        for file in os.listdir(path + '/' + filename[:-4]):
            data.append(extract(path + '/' + filename[:-4] + '/' + file))

    # shuffling data
    random.shuffle(data)        
    df = pd.DataFrame(data, columns = ['Mail','Class','Filename'])
    
    # adding classname for all files
    for index, row in df.iterrows():
        df.loc[index, 'Target'] = filenames.index(row['Class'] + '.zip')

    # generating csv
    df.to_excel('data_processed.xlsx',index = None)
    return os.getcwd().replace('\\','/') + '/data_processed.xlsx'
    
# function to process testing data and generate the CSV file in required format
# input: filenames and base path
# output: csv
def process_test(base, filename):
    data = []
    
    path = os.getcwd().replace('\\','/') + '/' + base 

    # unzipp file
    unzip(path + '/' + filename)

    # extract text from .msg file
    for file in os.listdir(path + '/' + filename[:-4]):
        data.append(extract(path + '/' + filename[:-4] + '/' + file))
        
    # shufflig data
    random.shuffle(data)   
     
    df = pd.DataFrame(data, columns = ['Mail','Target','Filename'])
    df.drop('Target',axis = 1,inplace = True)
    print('Dropping target')

    # generating csv
    df.to_excel('data_processed_test.xlsx',index = None)
    return os.getcwd().replace('\\','/') + '/data_processed_test.xlsx'


