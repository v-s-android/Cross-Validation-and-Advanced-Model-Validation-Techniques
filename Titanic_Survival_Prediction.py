"""
Introduction
Now that you have a feel for how to optimize your machine learning pipeline, let's practice with a real world dataset.
You'll use cross validation and a hyperparameter grid search to optimize your machine learning pipeline.

You will use the Titanic Survival Dataset to build a classification model to predict whether a passenger survived the sinking of the Titanic, based on attributes of each passenger in the data set.

You'll start with building a Random Forest Classifier, then modify your pipeline to use a Logistic Regression estimator instead. You'll evaluate and compare your results.
"""

"""
Objectives
After completing this lab you will be able to:

Use scikit-learn to build a model to solve a classification problem
Implement a pipeline to combine your preprocessing steps with a machine learning model
Interpret the results of your modelling
Update your pipeline with a different machine learning model
Compare the preformances of your classifiers
"""

!pip install numpy
!pip install matplotlib
!pip install pandas
!pip install scikit-learn
!pip install seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the Titanic dataset using Seaborn
titanic = sns.load_data('titanic')
titanic.head()


# Select relevant features and the target - for this you have to know the count() of each column
titanic.count()
"""
survived       891
pclass         891
sex            891
age            714
sibsp          891
parch          891
fare           891
embarked       889
class          891
who            891
adult_male     891
deck           203
embark_town    889
alive          891
alone          891
dtype: int64
"""
"""
Features to drop¶
deck has a lot of missing values so we'll drop it. age has quite a few missing values as well. Although it could be, 
embarked and embark_town don't seem relevant so we'll drop them as well. It's unclear what alive refers to so we'll ignore it.

Target
survived is our target class variable.
"""

# select feature and target class variable

features = ['pclass', 'sex', 'sibsp', 'parch', 'fare',  'class', 'who', 'adult_male', 'alone']
target = 'survived'

X = titanic[features]
y = titanic[target]

# Exercise 1. How balanced are the classes ? 
y.value_counts() 
"""
survived
0    549
1    342
Name: count, dtype: int64
So about 38% of the passengers in the data set survived.  
Because of this slight imbalance, we should stratify the data when performing train/test split and for cross-validation.
"""

# Exercise 2. Split the data into training and testing sets - Don't forget to consider imbalance in the target (y)
X_train, X_test, y_train, y_test = train_test_split(X, y , stratify = y, random_state = 10, test_size = 0.2)

# Define preprocessing transformers for numerical and categorical features
# Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features
numercial_features = X_train.select_dtypes(include = ['number']).columns.tolist() # ['pclass', 'age', 'sibsp', 'parch', 'fare']
categorical_features = X_train.select_dtypes(include = ['object', 'category']).columns.tolist() ['sex', 'class', 'who']

# Define separate preprocessing pipelines for both feature types - Use SimpleImputer, StandardScaler for numeric, OneHotEncoder for categoric values
numerical_transformer = Pipeline(
  steps = [
    ('imputer', SimpleImputer( strategy = 'median')),
    ('scaler' , StandardScaler())
  ]
)

categorical_transformer = Pipeline(
  steps = [
    ('imputer', SimpleImputer( straegy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
  ]
)

# Combine the transformers into a single column transformer
"""
We'll use the sklearn "column transformer" estimator to separately transform the features, which will then concatenate the output
as a single feature space, ready for input to a machine learning estimator.
"""
preprocessor = ColumnTransformer(
  transformers = [
    ('num', numerical_transformer, numercial_features ),
    ('cat', categorical_transformer, categorical_features)
  ]
)


"""
Create a model pipeline
Now let's complete the model pipeline by combining the preprocessing with a Random Forest classifier
"""

pipeline = Pipeline(
  [
    ('preprocessor' , preprocessor ),
    ('classifier', RandomForestClassifier(random_state = 10))
  ]
)

"""
Define a parameter grid
We'll use the grid in a cross validation search to optimize the model
"""
param_grid = {
      'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Perform grid search cross-validation and fit the best model to the training data
# Cross-validation method
cv = StratifiedKFold(n_splits = 5, shuffle=True)

# Exercise 3. Train the pipeline model
model = GridSearchCV(
  estimator = pipeline,
  cv = cv,
  param_grid = param_grid,
  scoring = 'acuracy',
  verbose = 2
)

model.fit(X_train, y_train)
"""
Fitting 5 folds for each of 12 candidates, totalling 60 fits
[CV] END classifier__max_depth=None, classifier__min_samples_split=2, classifier__n_estimators=50; total time=   0.1s
[CV] END classifier__max_depth=None, classifier__min_samples_split=2, classifier__n_estimators=50; total time=   0.1s
[CV] END classifier__max_depth=None, classifier__min_samples_split=2, classifier__n_estimators=50; total time=   0.1s
[CV] END classifier__max_depth=None, classifier__min_samples_split=2, classifier__n_estimators=50; total time=   0.1s
[CV] END classifier__max_depth=None, classifier__min_samples_split=2, classifier__n_estimators=50; total time=   0.1s
[CV] END classifier__max_depth=None, classifier__min_samples_split=2, classifier__n_estimators=100; total time=   0.2s
[CV] END classifier__max_depth=None, classifier__min_samples_split=2, classifier__n_estimators=100; total time=   0.2s
[CV] END classifier__max_depth=None, classifier__min_samples_split=2, classifier__n_estimators=100; total time=   0.2s
[CV] END classifier__max_depth=None, classifier__min_samples_split=2, classifier__n_estimators=100; total time=   0.2s
[CV] END classifier__max_depth=None, classifier__min_samples_split=2, classifier__n_estimators=100; total time=   0.2s
[CV] END classifier__max_depth=None, classifier__min_samples_split=5, classifier__n_estimators=50; total time=   0.1s
[CV] END classifier__max_depth=None, classifier__min_samples_split=5, classifier__n_estimators=50; total time=   0.1s
...
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=50; total time=   0.1s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=50; total time=   0.1s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=50; total time=   0.1s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=50; total time=   0.1s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=100; total time=   0.2s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=100; total time=   0.2s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=100; total time=   0.2s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=100; total time=   0.2s
[CV] END classifier__max_depth=20, classifier__min_samples_split=5, classifier__n_estimators=100; total time=   0.2s
"""

"""
Exercise 4. Get the model predictions from the grid search estimator on the unseen data
Also print a classification report
"""

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

"""
              precision    recall  f1-score   support

           0       0.83      0.87      0.85       110
           1       0.78      0.72      0.75        69

    accuracy                           0.82       179
   macro avg       0.81      0.80      0.80       179
weighted avg       0.81      0.82      0.81       179
"""
# Exercise 5. Plot the confusion matrix

conf_matrix = confusion_matrix(y_test , y_pred)

plt.figure()
sns.heatmap(conf_matrix , annot = True, cmap = "Blues", fmt='d')

plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

"""
Feature importances¶
Let's figure out how to get the feature importances of our overall model. You'll need to know how to do this for your final project.
First, to obtain the categorical feature importances, we have to work our way backward through the modelling pipeline to associate the feature importances
with their one-hot encoded input features that were transformed from the original categorical features.

We don't need to trace back through the pipeline for the numerical features, because we didn't transfrom them into new ones in any way.
Remember, we went from categorical features to one-hot encoded features, using the 'cat' column transformer.

Here's how you trace back through the trained model to access the one-hot encoded feature names:
"""

model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

"""
array(['sex_female', 'sex_male', 'class_First', 'class_Second',
       'class_Third', 'who_child', 'who_man', 'who_woman'], dtype=object)
Notice how the one-hot encoded features are named - for example, sex was split into two boolean features indicating whether the sex is male or female.

Great! Now let's get all of the feature importances and associate them with their transformed feature names.
"""

feature_importances = model.best_estimator_['classifier']

# Combine the numerical and one-hot encoded categorical feature names
feature_names = numerical_features + list(model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))

#Display the feature importances in a bar plot
#Define a feature importance DataFrame, then plot it

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

plt.figure()
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca.invert_yaxis()
plt.title('Most Important Features in predicting whether a passenger survived')
plt.xlabel('Importance Score')
plt.show()

#Print the test score
test_score = model.score(X_test, y_test)
print(f"test score :{test_score : .2f}") # 81.56%

