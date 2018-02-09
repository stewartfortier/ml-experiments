from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]] # [weight, texture]. weight is in kg. texture: 1 = "smooth", 0="bumpy"
labels = [0, 0, 1, 1] # 0 = "apple", 1 = "orange"

clf = tree.DecisionTreeClassifier() # we'll use a Decision Tree to classify objects
clf = clf.fit(features, labels) # fit() algorithm included in DecisionTreeClassifier class

new_fruit = [160, 0] # [160kg, "bumpy"]

output_key = {0: "apple", 1: "orange"} # translate results

prediction = clf.predict([new_fruit]) # see how classifier classifies new_fruit

print("Classifier's prediction: %s" % prediction) # show raw prediction

for item in prediction:
    print("Translation: %s" % output_key[item]) # translate to English!
