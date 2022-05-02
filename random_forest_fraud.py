# Packages / libraries
import os #provides functions for interacting with the operating system
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
fraud = pd.read_csv("C:/Users/HP/PycharmProjects/Excelrdatascience/Fraud_check_ran.csv")
fraud.head()
##Converting the Taxable income variable to bucketing.
fraud["income"]="<=30000"
fraud.loc[fraud["Taxable.Income"]>=30000,"income"]="Good"
fraud.loc[fraud["Taxable.Income"]<=30000,"income"]="Risky"
##Droping the Taxable income variable
fraud.drop(["Taxable.Income"],axis=1,inplace=True)
fraud.head()
fraud.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)
## As we are getting error as "ValueError: could not convert string to float: 'YES'".
## Model.fit doesnt not consider String. So, we encode
# Checking for null values
fraud.isnull().sum()
fraud.columns
# Limiting the data
fraud = fraud[['undergrad', 'marital', 'population', 'experience', 'urban', 'income']]

# Visualize the data using seaborn Pairplots
g = sns.pairplot(fraud, hue = 'income', diag_kws={'bw': 0.2})
# Investigate all the features by our y

features = ['undergrad', 'marital', 'population', 'experience', 'urban']


for f in features:
    plt.figure()
    ax = sns.countplot(x=f, data=fraud, hue = 'income', palette="Set1")
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in fraud.columns:
    if fraud[column_name].dtype == object:
        fraud[column_name] = le.fit_transform(fraud[column_name])
    else:
        pass
fraud.head()
# Your code goes here
X = fraud.drop('income', axis=1).values# Input features (attributes)
y = fraud['income'].values # Target vector
print('X shape: {}'.format(np.shape(X)))
print('y shape: {}'.format(np.shape(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, test_size=0.1, random_state=0)
# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
rf = RandomForestClassifier(n_estimators=100, criterion='entropy')
rf.fit(X_train, y_train)
prediction_test = rf.predict(X=X_test)

# source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# Accuracy on Test
print("Training Accuracy is: ", rf.score(X_train, y_train))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_test, y_test))

# Confusion Matrix
cm = confusion_matrix(y_test, prediction_test)
cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_norm, classes=rf.classes_)

#get all categorical columns
cat_columns = fraud.select_dtypes(['object']).columns

#convert all categorical columns to numeric
fraud[cat_columns] = fraud[cat_columns].apply(lambda x: pd.factorize(x)[0])
fraud.head()
##Splitting the data into featuers and labels
X = fraud.iloc[:,0:5]
y = fraud.iloc[:,5]
## Collecting the column names
colnames = list(fraud.columns)
predictors = colnames[0:5]
target = colnames[5]
##Splitting the data into train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,stratify = y)
##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)
model.oob_score_
##Predictions on train data
prediction = model.predict(x_train)
prediction
##Accuracy
# For accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
accuracy
##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)
##Prediction on test data
pred_test = model.predict(x_test)
##Accuracy
acc_test =accuracy_score(y_test,pred_test)

## In random forest we can plot a Decision tree present in Random forest
from sklearn.tree import export_graphviz
import pydotplus
from six import StringIO

tree = model.estimators_[5]

dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

## Creating pdf and png file the selected decision tree
graph.write_pdf('fraudrf.pdf')
graph.write_png('fraudrf.png')