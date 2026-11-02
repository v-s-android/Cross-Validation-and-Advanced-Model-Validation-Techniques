"""
In this lab

- Generate synthetic data for running targeted experiments using scikit-learn
- Train and evaluate a KNN classification model using a pipeline
- Tune model hyperparameters using a pipeline within a cross-validation grid search
- Build a more complex random forest classification pipeline using real-world data
- Extract the feature importances from the trained pipeline
"""

!pip install scikit-learn==1.6.0
!pip install matplotlib==3.9.3
!pip install seaborn==0.13.2

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix

"""
Train a model using a pipeline
We'll start with an example of building a pipeline, fitting it to the Iris data, and evaluating its accuracy.
"""

data = load_iris()
X, y = data.data , data.target
labels = data.target_names

"""
Instantiate a pipeline consisting of StandardScaler, PCA, and KNeighborsClassifier
"""
pipeline = Pipeline([
    ('scaler', StandardScaler()),       # Step 1: Standardize features
    ('pca', PCA(n_components=2),),       # Step 2: Reduce dimensions to 2 using PCA
    ('knn', KNeighborsClassifier(n_neighbors=5,))  # Step 3: K-Nearest Neighbors classifier
])

"""
Exercise 1. Split the data into training and test sets¶
Be sure to stratify the target.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 42 , stratify = y)
"""
X = Features (the input data used to make predictions).

y = Target (the output or answer the model needs to predict).

stratify=y - This ensures the training and testing sets maintain approximately the same proportions of each target class as the original dataset.

X_train and y_train → learn.
X_test and y_test → evaluate.
test_size=0.2 → 20% test data.
random_state=42 → repeatable split.
stratify=y → preserve target-class proportions.

Think of it like an exam: X_train and y_train are the practice material with answers, while X_test is the exam questions
and y_test is the answer key used to check how well the model performs.
"""

"""
Fit the pipeline on the training set
The pipeline consists of a sequence of three estimators, and should be viewed as the machine learning model we are training and testing.
Let's go ahead and fit the model to the training data and evaluate its accuracy.
"""

pipeline.fit(X_train, y_train)
# Measure the pipeline accuracy on the test data
test_score = pipeline.score(X_test, y_test)
print(f"{test_score : .3f}") # 0.900

# Get the model predictions
y_pred = pipeline.predict(X_test)

"""Exercise 2. Generate the confusion matrix for the KNN model and plot it"""

conf_matrix = confusion_matrix(y_test , y_pred)
"""
It compares two things:
y_test — the actual labels from your test dataset.
y_pred — the labels predicted by your model.
It counts how many predictions fall into each actual-versus-predicted category.
"""

plt.figure()
sns.heatmap( conf_matrix, annot=True, cmap = "Blues", fmt='d', xticklabels= labels, yticklabels= labels)
"""
annot=True → displays the numerical value inside each cell.
fmt='d' means integers (whole numbers), such as 10 or 25.
"""

# Set the title and labels
plt.title('Classification Pipeline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

"""
Instantiate the pipeline
We'll preprocess the data by scaling it and transforming it onto a to-be-determined number of principle components, follow that up with a KNN model classifier,
and combine these estimators into a pipeline.We'll then optimize the pipeline using crossvalidation over a hyperparameter grid search. This will allow us find the best model
for the set of trial hyperparamters.
"""

pipeline = Pipeline(
    [
        ('scalar', StandardScalar()),
        ('pca', PCA()),
        ('knn', KNeighborsClassifier())
    ]
)

#Define a model parameter grid to search over
# Hyperparameter search grid for numbers of PCA components and KNN neighbors
param_grid = {'pca__n_components': [2, 3],
              'knn__n_neighbors': [3, 5, 7]
             }
"""
Choose a cross validation method
To ensure the target is stratified, we can use scikit-learn's StratifiedKFold cross-validation class.
"""
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 10)


"""
Exercise 4. Determine the best parameters
Pass your pipeline, param_grid, and the StratifiedKFold cross validation method to GridSearchCV
"""
gird_search = GridSearchCV(
    estimator = pipeline,
    param_grid = param_grid,
    cv = cv,
    score = "accuracy",
    verbose=2
)

# Fit the best GridSearchCV model to the training data
grid_search.fit(X_train, y_train)

# Exercise 5. Evaluate the accuracy of the best model on the test set
new_score = grid_search.score(X_test, y_test)
print(f"{new_score : .3f}") # 0.933

# Display the best parameters
grid_search.best_params_  # {'knn__n_neighbors': 3, 'pca__n_components': 3}

# Plot the confusion matrix for the predictions on the test set
y_pred = grid_search.predict(X_test)

# Generate the confusion matrix for KNN
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
# Create a single plot for the confusion matrix
sns.heatmap(conf_matrix , annot=True, cmap = "Blues", xticklabels = labels , yticklabels = labels , fmt = d )

# Set the title and labels
plt.title('KNN Classification Testing Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()


    
