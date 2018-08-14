import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import pydotplus
import collections


data = pd.read_csv("/Users/rupengda/Downloads/data clean with Bean Type.csv")
data["Cocoa Percent"]=data["Cocoa Percent"].str.strip("%").astype(float)/100
data.info()

# DT with  Cocoa Percent, Bean Origin, Company Location
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# categorical data
X_C_CL_P = LabelEncoder()
X_C_CL=X_C_CL_P.fit_transform(data["Company Location"])
X_C_BO_P=LabelEncoder()
X_C_BO=X_C_BO_P.fit_transform(data["Bean Origin"].astype(str))
X_C_BT_P=LabelEncoder()
X_C_BT=X_C_BT_P.fit_transform(data["Bean Type"].astype(str))



X_N_CP=data["Cocoa Percent"].values
Y_C_R_P= LabelEncoder()
Y_C_R=Y_C_R_P.fit_transform(data["Rating"])
Y_C_R=Y_C_R.reshape(len(Y_C_R),1)
#creat test dataset
x_list=(X_N_CP,X_C_CL,X_C_BO,X_C_BT)
X=np.vstack(x_list).T

X_train, X_test, y_train, y_test = train_test_split(X, Y_C_R, test_size=0.01, random_state=400)


x1_list=(X_N_CP,X_C_CL,X_C_BO)
X1=np.vstack(x1_list).T

X1_train, X1_test, y_train, y_test = train_test_split(X1, Y_C_R, test_size=0.01, random_state=400)

#%%-----------------------------------------------------------------------
# perform training with entropy.
# Decision tree with entropy
clf_entropy = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy1 = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
clf_entropy.fit(X_train, y_train)
clf_entropy1.fit(X1_train, y_train)
#%%-----------------------------------------------------------------------


# display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser


N_list=["Coca Percent","Company Location","Bean Origin","Bean Type"]
dot_data = export_graphviz(clf_entropy, filled=True, rounded=True,  feature_names=N_list, out_file=None, class_names=Y_C_R_P.classes_.astype(str))

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decisiontree_entropy.pdf")
webbrowser.open_new(r'decisiontree_entropy.pdf')

y_pred = clf_entropy.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")



N1_list=["Coca Percent","Company Location","Bean Origin"]
dot_data1 = export_graphviz(clf_entropy1, filled=True, rounded=True,  feature_names=N1_list, out_file=None, class_names=Y_C_R_P.classes_.astype(str))

graph = graph_from_dot_data(dot_data1)
graph.write_pdf("decisiontree_entropy1.pdf")
webbrowser.open_new(r'decisiontree_entropy1.pdf')

y_pred1 = clf_entropy1.predict(X1_test)
print("Accuracy : ", accuracy_score(y_test, y_pred1) * 100)
print("\n")


print ('-'*40 + 'End Console' + '-'*40 + '\n')
