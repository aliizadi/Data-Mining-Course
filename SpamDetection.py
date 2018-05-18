import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import  MultinomialNB
import string
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.cross_validation import cross_val_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/spam.csv', encoding='latin-1')

# print(df.head())

df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

text_feat = df['v2'].copy()


def text_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    return " ".join(text)


text_feat = text_feat.apply(text_process)

vectorizer = TfidfVectorizer("english")
# vectorizer = CountVectorizer("english")

features = vectorizer.fit_transform(text_feat)

X_train, X_test, y_train, y_test = train_test_split(features, df["v1"], test_size=0.2, random_state=111)

model = MultinomialNB(alpha=0.2)

accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print('Accuracy', np.mean(accuracy), accuracy)

lb = LabelBinarizer()
y_train = np.array([number[0] for number in lb.fit_transform(y_train)])

recall = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
print('Recall', np.mean(recall), recall)
precision = cross_val_score(model, X_train, y_train, cv=5, scoring='precision')
print('Precision', np.mean(precision), precision)



