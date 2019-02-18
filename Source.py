
# importing the numpy library and setting it as np
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

#This will import the python time and system features
import time
import sys
#imports the sleep feature to slow down prints by 3 seconds
from time import sleep

print("Welcome to the flower database")
print("This program will list three flowers")
print("along with their features and the label")
print("Start program")
sleep(3)

# Import the iris data set
iris_data_set = load_iris()

# Print out the features in the data set
print(" " * 40)
print("Features")
print(iris_data_set.feature_names)

# Print out the data set's flower
print(" " * 40)
print("Labels")
print(iris_data_set.target_names)

print(" " * 40)
print("Notice the data below will line up perfectly with")
print("the features above. This is feature data")
print(iris_data_set.data[0])

print(" " * 40)
print("Show the label for now 0. We are just taking a peek at the data.")
print("Label Table")
print("0 = setosa 1 = versicolor, 2 = virginica")
print('Label =', iris_data_set.target[0])

print(" " * 40)
print("The iris dataset will be used to classify 3 types of flowers")
print("The dataset is 150 rows, 50 rows for each flower")
print("These rows are in order.")
print("Rows 0-49 = setosa")
print("Rows 50-99 = versicolor")
print("Rows 110-149 = virginica")

#Print out the full data set to have a reference
print(" " * 40)
print("The full dtat set to reference")
for i in range(len(iris_data_set.target)):
    print("Example %d: Label %s: Features %s:" % (i, iris_data_set.target[i], iris_data_set.data[i]))

# Remove one type of each flower: type
# Because we are going
# This test data will be data never before by our classifier
test_index = [0, 50, 100]

# Now lets make some training data
# This is the bulk of our data
# We will have 147 rows of data to use for training

train_target = np.delete(iris_data_set.target, test_index)
train_data = np.delete(iris_data_set.data, test_index, axis=0)

# Here we go back to our testing data
# This is unseen data by our classifier
# This contains only 3 test flowers

test_target = iris_data_set.target[test_index]
test_data = iris_data_set.data[test_index]

# Create our classifier
# We will be using a decision tree
dt_clf = tree.DecisionTreeClassifier()
dt_clf.fit(train_data, train_target)

# Here we classify the data
print(" " * 40)
print("Test Data")
print(test_target)
print(" " * 40)
print("Machine's prediction data, check against test")
print("Is this a match?")
print(dt_clf. predict(test_data))

