#ALGORITHM 1
# Decision Tree : ex= personalized ads in mobile apps. 
# it is used to classify the samples based on the features of the user and predict whether the user will click on the ad or not.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Other sklearn modules
from sklearn import datasets
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.inspection import DecisionBoundaryDisplay

#Get Iris Data
Iris_dataset = datasets.load_iris()

X = Iris_dataset.data[:,:2]
y = Iris_dataset.target
Decision_tree_model = tree.DecisionTreeClassifier()
Decision_tree_model.fit(X,y)

#Predict species for new samples (sepal length, sepal width).
new_data = np.array([
	[5.1, 3.5],
	[6.7, 3.0],
])
predicted_classes = Decision_tree_model.predict(new_data)
predicted_species = Iris_dataset.target_names[predicted_classes]

for sample, species in zip(new_data, predicted_species):
	print(f"New data {sample} -> predicted species: {species}")

plt.figure()
plot_tree(Decision_tree_model)
plt.savefig("Decision Tree model.svg")


plt.figure()
# Plot decision boundary
ax = plt.subplot()
display = DecisionBoundaryDisplay.from_estimator(
Decision_tree_model,
X,
response_method="predict",
cmap="coolwarm",
alpha=0.8,
ax=ax)
for class_index, species in enumerate(Iris_dataset.target_names):
	class_points = X[y == class_index]
	ax.scatter(
		class_points[:, 0],
		class_points[:, 1],
		c=np.full(class_points.shape[0], class_index),
		cmap="coolwarm",
		vmin=0,
		vmax=len(Iris_dataset.target_names) - 1,
		label=species,
	)
ax.legend(title="Species")
plt.title('Decision Tree Classifier')
# plt.show()
plt.savefig("Decision Tree Classifier.svg")


#ALGORITHM 2
# Random Forest : extension of decision tree. a part of Ensemble learning in sklearn. 
# Random forest is a big decision tree and it has multiple tress within it. it has one huge branch and have smaller branches within it.
# an estimator and a random number of decision trees are created and the final output is based on the majority of the outputs of all the decision trees.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# #Other sklearn modules
from sklearn import datasets
from sklearn import ensemble
from sklearn.inspection import DecisionBoundaryDisplay

# #Get Iris Data
Iris_dataset = datasets.load_iris()

X = Iris_dataset.data[:,:2]
y = Iris_dataset.target

Random_Forest_model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
Random_Forest_model.fit(X, y)

# #Predict species for new samples (sepal length, sepal width).
new_data = np.array([
	[5.1, 3.5],
	[6.7, 3.0],
])
predicted_classes = Random_Forest_model.predict(new_data)
predicted_species = Iris_dataset.target_names[predicted_classes]

for sample, species in zip(new_data, predicted_species):
	print(f"New data {sample} -> predicted species: {species}")

plt.figure()
plot_tree(Random_Forest_model)
plt.savefig("Random Forest model.svg")


plt.figure()
# Plot decision boundary
ax = plt.subplot()
display = DecisionBoundaryDisplay.from_estimator(
Random_Forest_model,
X,
response_method="predict",
cmap="coolwarm",
alpha=0.8,
ax=ax)
for class_index, species in enumerate(Iris_dataset.target_names):
	class_points = X[y == class_index]
	ax.scatter(
		class_points[:, 0],
		class_points[:, 1],
		c=np.full(class_points.shape[0], class_index),
		cmap="coolwarm",
		vmin=0,
		vmax=len(Iris_dataset.target_names) - 1,
		label=species,
	)
ax.legend(title="Species")
plt.title('Random Forest Classifier')
# plt.show()
plt.savefig("Random Forest Classifier.svg")

#ALGORITHM 3
# KNN : K nearest neighbor : it is a supervised learning algorithm. it is used for classification and regression. it is based on the principle that similar things exist in close proximity.
# K says if we want to consider the nearest 3 neighbors or 5 neighbors or 7 neighbors. it is based on the distance between the points.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

#Get Iris Data
Iris_dataset = datasets.load_iris()

X = Iris_dataset.data[:,:2]
y = Iris_dataset.target

KNN_model = KNeighborsClassifier(n_neighbors=10)
KNN_model.fit(X, y)

#Predict species for new samples (sepal length, sepal width).
new_data = np.array([
	[5.1, 3.5],
	[6.7, 3.0],
])
predicted_classes = KNN_model.predict(new_data)
predicted_species = Iris_dataset.target_names[predicted_classes]

for sample, species in zip(new_data, predicted_species):
	print(f"New data {sample} -> predicted species: {species}")


plt.figure()
# Plot decision boundary
ax = plt.subplot()
display = DecisionBoundaryDisplay.from_estimator(
KNN_model,
X,
response_method="predict",
cmap="coolwarm",
alpha=0.8,
ax=ax)
for class_index, species in enumerate(Iris_dataset.target_names):
	class_points = X[y == class_index]
	ax.scatter(
		class_points[:, 0],
		class_points[:, 1],
		c=np.full(class_points.shape[0], class_index),
		cmap="coolwarm",
		vmin=0,
		vmax=len(Iris_dataset.target_names) - 1,
		label=species,
	)
ax.legend(title="Species")
plt.title('K-Nearest Neighbors Classifier n=10')
# plt.show()
plt.savefig("K-Nearest Neighbors Classifier_10.svg")




#ALGORITHM 4
# R nearest neighbor : it is a Radius neighbour algorithm. it considers all the points within a certain radius. 


#Classifiers
import numpy as np
import matplotlib.pyplot as plt

# #different models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# #Other sklearn modules
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay

Iris_dataset = datasets.load_iris()
X = Iris_dataset.data[:,:2]
y = Iris_dataset.target

# #Split the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

# #Logistic regression
Logistic_model = LogisticRegression()
Logistic_model.fit(X_train,y_train)

# # Plot decision boundary
ax = plt.subplot()
display = DecisionBoundaryDisplay.from_estimator(
Logistic_model,
X,
response_method="predict",
cmap="coolwarm",
alpha=0.8,
ax=ax)
plt.scatter(Iris_dataset.data[:, 0], Iris_dataset.data[:, 1], c=Iris_dataset.target,cmap="coolwarm")
plt.title('Logistic Regression')
plt.show()

# #Support Vector machines
kernels = ['linear','poly','rbf','sigmoid']

for kernel in kernels:
    SVC_model = SVC(kernel=kernel)
    SVC_model.fit(X_train,y_train)

    # Plot decision boundary
    ax = plt.subplot()
    display = DecisionBoundaryDisplay.from_estimator(
    SVC_model,
    X,
    response_method="predict",
    cmap="coolwarm",
    alpha=0.8,
    ax=ax,
    )
    plt.title(f"{kernel} kernel")
    plt.scatter(Iris_dataset.data[:, 0], Iris_dataset.data[:, 1], c=Iris_dataset.target,cmap="coolwarm")
    plt.show()