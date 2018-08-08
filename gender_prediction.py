'''
author     = "Priagung Khusumanegara"
copyright  = "Copyright 2018, Gender Prediction"
credits    = ["nltk"]
license    = "GPL"
version    = "1.0.0"
maintainer = "Prigung Khusumanegara"
email      = "priagung.123@gmail.com"
status     = "Development"
'''

import nltk
import random
import pandas as pd

#function to extract the features from name
def extract_features(name):
    return {
        'last_char': name[-1],
        'last_two': name[-2:],
        'last_three': name[-3:],
        'first': name[0],
        'first2': name[:1]
    }

#training and Testing data
f_names = nltk.corpus.names.words('female.txt')
m_names = nltk.corpus.names.words('male.txt')

#Labeling based on file name
all_names = [(i, 'Male') for i in m_names] + [(i, 'Female') for i in f_names]
random.shuffle(all_names)

#split the number of training and testing data
test_set = all_names[500:]
train_set= all_names[:500]

#extract the features from training and testing data
test_set_feat = [(extract_features(n), g) for (n, g) in test_set]
train_set_feat= [(extract_features(n), g) for (n, g) in train_set]

#apply nltk NaiveBayesClassifier to training data
classifier = nltk.NaiveBayesClassifier.train(train_set_feat)

#load data
df_rp = pd.read_excel("RPUser.xlsx")

#take out data that have null value in FinalName column
df_rp_pre1 = df_rp[df_rp.FinalName.notnull()]
df_rp_pre1['LowName'] = df_rp_pre1['FinalName'].str.lower()

df_rp_pre1.to_excel('df_rp_pre1.xlsx')

#convert to list 
data_pred_list = df_rp_pre1['LowName'].tolist()

#apply model to data
result = []
for name in data_pred_list:
    guess = classifier.classify(extract_features(name))
    result.append([name,guess])

#list to dataframe
df_result = pd.DataFrame(result, columns=["Name","Gender"])

#export results to excel
df_result.to_excel('df_result.xlsx')