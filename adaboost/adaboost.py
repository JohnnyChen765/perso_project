from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import helper
import pandas as pd

# Load data
n_individus = 100
X = pd.concat([helper.orc_generator(n_individus), helper.human_generator(n_individus), helper.elf_generator(n_individus)])
X = shuffle(X)
y = X['race'].values
X.drop(columns = 'race', inplace=True)
# import ipdb; ipdb.set_trace()

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target


# il est possible de mettre des string dans y, au lieu de juste 0, 1... 
# le modèle va inférer ces classes

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

# Create adaboost classifer object
classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)
# Train Adaboost Classifer
classifier.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = classifier.predict(X_test)

# print(classifier.score(X_train, y_train))
#     # is the same as
# print(metrics.accuracy_score(classifier.predict(X_train), y_train))

results = pd.DataFrame({'predict': y_pred, 'real': y_test})
print(results)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
