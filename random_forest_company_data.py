#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
company = pd.read_csv("C:/Users/HP/PycharmProjects/Excelrdatascience/Company_Data_ran.csv")
company.head()
##Looking into unique value
company["Sales"].unique()
##Preforming how many times each number is repeated
company["Sales"].value_counts()
##Looking into median to check the median-- middle value, which can help us in Stratified sampling
np.median(company["Sales"])
company["sales"]="<=7.49"
company.loc[company["Sales"]>=7.49,"sales"]=">=7.49"
company.drop(["Sales"],axis=1,inplace=True)
company.head()
#get all categorical columns
cat_columns = company.select_dtypes(['object']).columns

#convert all categorical columns to numeric
company[cat_columns] = company[cat_columns].apply(lambda x: pd.factorize(x)[0])
company.head()
company.columns
# Limiting the data
company2 =company[['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US', 'sales']]

# Visualize the data using seaborn Pairplots
g = sns.pairplot(company2, hue = 'sales', diag_kws={'bw': 0.2})
# Investigate all the features by our y

features = ['CompPrice', 'Income', 'Advertising', 'Population', 'ShelveLoc', 'Age',
       'Education','Urban','US']


for f in features:
    plt.figure()
    ax = sns.countplot(x=f, data=company,hue='sales', palette="Set1")
print(company)
from sklearn.preprocessing import MinMaxScaler
# Scaling our columns

scale_vars = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
         'Age', 'Education']
scaler = MinMaxScaler()
company[scale_vars] = scaler.fit_transform(company[scale_vars])
company.head()
# Your code goes here
X = company.iloc[:,0:10].values# Input features (attributes)
y = company['sales'].values # Target vector

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
from sklearn.ensemble import RandomForestClassifier
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



