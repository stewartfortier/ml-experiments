import numpy as np
import matplotlib.pyplot as plt

# * Importance of Features *
# The features we choose for our data largely determine the accuracy of our classifier
# If we feed it features with little or no predictive value, it can't learn much
# Visualizing the distribution of features can be a great way for us to assess
# if a feature is impactful enough to include in our training data.



# * Example *
# Let's imagine that we'd like to classify a dog as a greyhound or a lab
# We decide to look at two features: height and eye color
# Greyhounds are, on average, taller than labs
# Eye color, however, appears to be distributed randomly

# create population of dogs
greyhounds = 500
labs = 500

# on average, greyhound height = 28 inches, lab height = 24 in
# and variance in height (+-4 inches) is evenly distributed
grey_height = 28 + 4 * np.random.randn(greyhounds) # creates a numpy array of 500 heights distributed evenly
lab_height = 24 + 4 * np.random.randn(labs)

# create histogram with heights
plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])

# visualize grouping of height, will help us understand usefulness of height
plt.show() # we can see a correlation between height and breed

# If eye color was random, we would see something like a 50/50 split amongst breeds
# Including this feature in our training data might actually *hurt* classifier performance




# /* Rules of Thumb for Selecting Features *

# 1) Informative.
# - Avoid non-correlated features, like eye color in the example above.
# 2) Independent.
# - Avoid reduntant features.
# - For example, Height in cm AND height in inches. Our classifier doesn't realize they're the same thing.
# 3) Simple.
# - Opt for features that are *clear* and *simple*
# - It's easier to learn how distance affects the time to travel somewhere
# - As opposed to two geo-coordinates, which would require learning how geo-coordinates relate to each other.
