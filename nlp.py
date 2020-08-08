import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score,confusion_matrix

df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
print(df.head())
print(df['Review'])

ps=PorterStemmer()
c=[]
reviews=re.sub('[^a-zA-Z]',' ',df['Review'][0])
reviews=reviews.lower()
print(reviews)
for i in range(0,1000):
    reviews=re.sub('[^a-zA-Z]',' ',df['Review'][i])
    reviews=reviews.lower()
    reviews=reviews.split()
    reviews=[ps.stem(word) for word in reviews if not word in set(stopwords.words('english'))]
    reviews=' '.join(reviews)
    c.append(reviews)
print(c[0:5])

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(c).toarray()
y=df.iloc[:,-1].values
print(x[0:5])
print(y[0:5])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
modelNB = MultinomialNB()
modelNB.fit(x_train, y_train)
y_predNB = modelNB.predict(x_test)
y_predNB = (y_predNB > 0.5)
print(y_predNB[0:5])
print(y_test[0:5])

print("Confusion matrix for Naive Bayes model: ",confusion_matrix(y_test,y_predNB))
print("Accuracy score for Naive Bayes model: ",accuracy_score(y_test,y_predNB)*100)

#Using Deep learning
modelNN= Sequential([
    Dense(1500,activation='sigmoid'),
    Dense(1000,activation='sigmoid'),
    Dense(100,activation='sigmoid'),
    Dense(1,activation='sigmoid')
])
modelNN.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
modelNN.fit(x_train,y_train,epochs=50,batch_size=10)
y_predNN = modelNN.predict(x_test)
y_predNN = (y_predNN > 0.5)
print(y_predNN[0:5])
print(y_test[0:5])

print("Confusion matrix for Deep learning model: ",confusion_matrix(y_test,y_predNN))
print("Accuracy score for Deep learning model: ",accuracy_score(y_test,y_predNN) * 100)
