# classifier.py: building a custom k-Nearest Neighbor classifier.
import random
from scipy.spatial import distance


# * Custom Classifier Class *

class ScrappyKNN():
    def fit(self, training_data, training_labels):
        # store data & labels in classifier
        self.training_data = training_data
        self.training_labels = training_labels

    def predict(self, test_data):
        predictions = []
        for row in test_data:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_index = 0
        best_distance = euc(row, self.training_data[best_index])

        for i in range(1, len(self.training_data)):
            dist = euc(row, self.training_data[i])
            if dist < best_distance:
                best_distance = dist
                best_index = i

        return self.training_labels[best_index]


# * Helper Functions *

def euc(a, b):
    return distance.euclidean(a,b)

def print_predicted_and_actual(predicted, test_data):
    for i, val in enumerate(test_data):
        prediction = predicted[i]
        print('Predicted: %s. Actual: %s.' % (prediction, val))
        if not prediction == val:
            print('Incorrect prediction!')


# * Create Training & Test Datasets *

# import dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

# Split data into train and test data using sklearn's train_test_split class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


# * Create a Classifier *

my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)


# * Assess Accuracy of Classifier *

# print_predicted_and_actual(predictions, y_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy score: %s" % accuracy)


# * Notes *
# Pros of KNN
# - Relatively simple
# Cons:
# - Computationally intensive (requires iterating through all training data for each prediction)
# - Hard to represent relationships between features (some features might be more informative than others, but further away)
