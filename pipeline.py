# pipeline.py: a simple comparison between different classifiers

# * Helper Functions *

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

# Create decision tree classifier...
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

# ...or a k-nearest neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)


# * Assess Accuracy of Classifier *

print_predicted_and_actual(predictions, y_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy score: %s" % accuracy)
