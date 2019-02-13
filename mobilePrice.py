

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, 20].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, y_train)

# Predicting the Test set results
y_predLR = classifierLR.predict(X_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cmLR = confusion_matrix(y_test, y_predLR)

#K Nearest neighbours
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)

y_predKNN = classifierKNN.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmKNN = confusion_matrix(y_test, y_predKNN)
"""
confusion matrix
each row stands for actual class and each column for predicted class 
"""

#naive bayes
from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB();
classifierNB.fit(X_train, y_train)

y_predNB = classifierNB.predict(X_test)

cmNB = confusion_matrix(y_test , y_predNB)

#support vector machine
from sklearn.svm import SVC
classifierSVC = SVC(kernel = 'linear' , random_state = 0)
classifierSVC.fit(X_train , y_train)

y_predSVC = classifierSVC.predict(X_test)
cmSVC = confusion_matrix(y_test , y_predSVC)
 
#decision tree model
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier( criterion = "entropy",random_state=0)
classifierDT.fit(X_train,y_train)

y_predDT = classifierDT.predict(X_test)
cmDT = confusion_matrix(y_test , y_predDT)
