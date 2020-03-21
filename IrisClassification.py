import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Preproccessing
dataset = pd.read_csv('iris.data')
le = preprocessing.LabelEncoder()
sepal_length = le.fit_transform(list(dataset['sepal_length']))
sepal_width = le.fit_transform(list(dataset['sepal_width']))
petal_length = le.fit_transform(list(dataset['petal_length']))
petal_width = le.fit_transform(list(dataset['petal_width']))
Class = le.fit_transform(list(dataset['class']))

predict = Class

x = list(zip(sepal_length, sepal_width, petal_length, petal_width))
y = list(predict)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 0) 

#Model
model = KNeighborsClassifier(n_neighbors = 9)
model.fit(x_train, y_train)

def specieindex(num1):
	if num1 == 0:
		return 'Iris-setosa'
	elif num1 == 1:
		return 'Iris-versicolor'
	elif num1 == 2:
		return 'Iris-virginica'


#Prediction
prediction = model.predict(x_test)
for i in range(len(prediction)):

	print('Actual : ', specieindex(prediction[i]), 'Prediction ', specieindex(prediction[i]))

#Accuracy
print('The accuracy of the model is : ', (model.score(x_train ,y_train))*100, '%')