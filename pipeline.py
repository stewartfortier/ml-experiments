# Let's compare different types of classifiers!!

# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

# Split data into train and test data using sklearn's train_test_split class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Create decision tree classifier...
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

# ...or a k-nearest neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

'''
# Optional: print predicted vs. actual values...
for i, val in enumerate(y_test):
    prediction = predictions[i]
    print('Predicted: %s. Actual: %s.' % (prediction, val))
    if not prediction == val:
        print('Incorrect prediction!')
'''

# print accuracy score of decision tree classifier...
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy score: %s" % accuracy)
