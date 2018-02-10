import numpy as np
from sklearn.datasets import load_iris # SKLearn includes a few datasets
from sklearn import tree
import random

iris = load_iris()
test_idx = [0, 50, 100] # The training data has 50 examples of each flower, so we'll remove one example of each as test data

'''
Overview of dataset attributes:
iris.feature_names -> array of feature names
iris.target_names -> array of labels
iris.data -> all feature data (ex. [[5.1, 3.5, 1.4], [3.4, 2.1, .7], .... , [4.6, 3.0, 2.3]])
iris.target -> all corresponding labels (ex. [0, 2, 1, ..., 0])
iris.data[0] -> first row of dataset (ex. [5.1, 3.5, 1.4])
iris.target[0] -> label for first row of data (ex. 0)
'''

# * Train Classifier and Make Predictions *

# create training data...
# by removing rows at test indices
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# create testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# create classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print('Test target: %s' % test_target)
print('Test data: %s' % test_data)
print('Feeding test data to classifier...')

prediction = clf.predict(test_data)

# print('Prediction: %s' % prediction)


# * Visualize Decision Tree *
# Creates a pdf of  decision tree and saves to current directory
import graphviz

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")



# * Interpreting a Decision Tree *
idx_of_test_data = random.randint(0, len(test_data) - 1) # choose a random test row for quiz

print("QUIZ TIME")
print("Open up iris.pdf and answer the following...")
print("Given the following test data:")
print(iris.feature_names)
print(test_data[idx_of_test_data])
print("What type of flower will our classifier predict?")

incorrect = True
while incorrect:
    for i, val in enumerate(iris.target_names):
        print('%s: %s' % (i, val))
    user_input = int(input("Answer: "))

    if user_input == prediction[idx_of_test_data]:
        print("Correct, nice work!")
        incorrect = False
    else:
        print("Incorrect, try again!")
