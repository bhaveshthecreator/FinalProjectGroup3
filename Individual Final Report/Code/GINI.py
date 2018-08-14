import pandas as pd
import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\bhave\\Anaconda3\\Library\\bin\\graphviz'
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
import collections

data = pd.read_csv("data clean with Bean Type.csv")
data["Cocoa Percent"]=data["Cocoa Percent"].str.strip("%").astype(float)/100
data.info()

print (data.isnull().sum())
u=data["Bean Type"].notnull()
data=data[u]
print (data.isnull().sum())

# categorical data
X_C_CL_P = LabelEncoder()
X_C_CL=X_C_CL_P.fit_transform(data["Company Location"])
X_C_BO_P=LabelEncoder()
X_C_BO=X_C_BO_P.fit_transform(data["Bean Origin"].astype(str))
X_C_BT_P = LabelEncoder()
X_C_BT=X_C_BT_P.fit_transform(data["Bean Type"].astype(str))

X_N_CP=data["Cocoa Percent"].values
Y_C_R_P= LabelEncoder()
Y_C_R=Y_C_R_P.fit_transform(data["Rating"])
Y_C_R=Y_C_R.reshape(len(Y_C_R),1)
#creat test dataset
x_list=(X_N_CP,X_C_CL,X_C_BO,X_C_BT)
X=np.vstack(x_list).T

X_train, X_test, y_train, y_test = train_test_split(X, Y_C_R, test_size=0.01, random_state=400)

#model set up
clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
clf = clf.fit(X_train,y_train)

N_list=["Coca Percent","Company Location","Bean Origin","Bean Type"]
class_name=data["Rating"].value_counts()
dot_data = tree.export_graphviz(clf,
                                feature_names=N_list,
                                out_file=None,
                                filled=True,
                                rounded=True,
                                class_names=Y_C_R_P.classes_.astype(str)
                                )
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
graph.write_svg('tree.svg')

graph.write_pdf("decision_tree_gini.pdf")

# calculate metrics gini model
y_pred_gini = clf.predict(X_test)
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print ('-'*80 + '\n')


#----find out bean location of our target (BO<5.5)
BO_target=np.arange(6)
BO_F=X_C_BO_P.inverse_transform(BO_target)