import os
import pandas as pd
import numpy as np
import nltk
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier 

from sklearn.svm import LinearSVC

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

nltk.download('stopwords')
nltk.download('wordnet')
data_path = "./datasets/bbc/"
os.listdir(data_path)
folders = [f for f in os.listdir(data_path) if f not in ["README.TXT", "bbc.csv"]]

news = []
category = []
for folder in folders:
    internal_path = data_path + folder
    files = os.listdir(internal_path)
    for t_files in files:
        t_path = internal_path + '/' + t_files
        with open(t_path, 'r') as f:
            content = f.readlines()
        content = ' '.join(content)
        news.append(content)
        category.append(folder)
tempdict = {'News' :news, 'Category': category} 
df = pd.DataFrame(tempdict) 
df.to_csv("./datasets/bbc.csv")
lem = WordNetLemmatizer()

processed_text = []
new_text = " "
for n in range(len(df.News)):
    new_text = re.sub(r"\W", " ", str(df.News[n])) 
    new_text = new_text.lower() 
    new_text = re.sub(r"\s+[a-zA-Z]\s+", " ", new_text) 
    new_text = re.sub(r"\s+", " ", new_text) 
    processed_text.append(new_text) 

processed = map(lambda x:' '.join([lem.lemmatize(word) for word in x.split()]), processed_text) 
processed_text = list(processed)

stopwords = nltk.corpus.stopwords.words("english")
count = CountVectorizer(min_df = 5, max_df=0.6, stop_words=stopwords)
edit_text_1 = count.fit_transform(processed_text).toarray()
edit_text_1 = SelectKBest(chi2, k=1500).fit_transform(edit_text_1,df.Category)

def get_scores_of(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te) 
    Acc = metrics.accuracy_score(y_te,y_pred)
    F1 = metrics.f1_score(y_te,y_pred,average='macro')
    Pre = metrics.precision_score(y_te,y_pred, average='macro')
    Rec = metrics.recall_score(y_te,y_pred, average='macro')
    return Acc, F1, Pre, Rec

K = StratifiedKFold(n_splits=10) 
ModelScoresAcc, ModelScoresF1, ModelScoresPre, ModelScoresRec = [],[],[],[]
tfidf = TfidfVectorizer(min_df=3, stop_words=stopwords, norm='l2', ngram_range=(1,1))
edit_text_2 = tfidf.fit_transform(processed_text).toarray() 
edit_text_2 = SelectKBest(chi2, k=1500).fit_transform(edit_text_2,df.Category)
tfidf = TfidfVectorizer(min_df=3, stop_words=stopwords, norm='l2', ngram_range=(2,2))
edit_text_3 = tfidf.fit_transform(processed_text).toarray() 
edit_text_3 = SelectKBest(chi2, k=1500).fit_transform(edit_text_3,df.Category)
edit_text = np.hstack((edit_text_1, edit_text_2, edit_text_3))
for train_i, test_i in K.split(edit_text,category):
    X_train, X_test, y_train, y_test = edit_text[train_i], edit_text[test_i], df.Category[train_i], df.Category[test_i]
    acc, f1, pre, rec = get_scores_of(LinearSVC(max_iter=6000, multi_class='ovr'),X_train, X_test, y_train, y_test)
    ModelScoresAcc.append(acc)
    ModelScoresF1.append(f1)
    ModelScoresPre.append(pre)
    ModelScoresRec.append(rec)

print("With LinearSVC:")
print("Accuracy of the model: {:.2f}".format(float(np.mean(ModelScoresAcc)*100)), "%")
print("Macro averaged F1 score of the model: {:.2f}".format(float(np.mean(ModelScoresF1)*100)), "%")
print("Macro averaged precision of the model: {:.2f}".format(float(np.mean(ModelScoresPre)*100)), "%")
print("Macro averaged Recall of the model: {:.2f}".format(float(np.mean(ModelScoresRec)*100)), "%")