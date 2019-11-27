import helper
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

number_of_points = 100
first_circle = helper.random_circle(number_of_points, y_offset = 2)
first_circle['classe'] = [0 for i in range(0, number_of_points)]
second_circle = helper.random_circle(number_of_points, x_offset = 2)
second_circle['classe'] = [1 for i in range(0, number_of_points)]


dataset = pd.concat([first_circle, second_circle])
def get_color_for_classe(class_array):
    color_bank = ['r', 'b']
    color = [color_bank[classe] for classe in class_array]
    return color

# plt.plot(dataset['x'], dataset['y'], 'ro', markersize=4)
# plt.scatter(dataset['x'], dataset['y'], c=color)
# plt.grid()

x_train, x_test, y_train, y_test = train_test_split(dataset[['x', 'y']], dataset['classe'], train_size=0.8)
clf = svm.SVC(gamma='scale')
clf.fit(x_train, y_train)

# support_vector_indexes = clf.support_
# support_vectors = clf.support_vectors_
# dual_coef = clf.dual_coef_ #les yi*alpha_i
# decision_function = clf.decision_function

x_plot = helper.random_points(1000, -5, 5)
y_plot = helper.random_points(1000, -5, 5)

y_predicted = clf.predict(x_test)
mse = helper.mse(y_predicted, y_test)

plt.scatter(x_test['x'], x_test['y'], c=get_color_for_classe(y_predicted))
# plt.legend(mse)
plt.grid()

plt.plot(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], 'go')
plt.show()

