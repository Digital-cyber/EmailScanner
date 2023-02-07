#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#import emails
emails = pd.read_csv('emails.csv')

#clean the data
corpus = []
for i in range(0, len(emails)):
    email = re.sub('[^a-zA-Z]', ' ', emails['email'][i])
    email = email.lower()
    email = email.split()
    ps = PorterStemmer()
    email = [ps.stem(word) for word in email if not word in set(stopwords.words('english'))]
    email = ' '.join(email)
    corpus.append(email)

#Create the Bag of Words Model
cv = CountVectorizer(max_features = 500)
X = cv.fit_transform(corpus).toarray()
y = emails.iloc[:, 1].values

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#train the model
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#test the model
y_pred = classifier.predict(X_test)

#evaluate the model
cm = confusion_matrix(y_test, y_pred)
print(cm)
