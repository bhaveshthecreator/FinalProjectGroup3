import pandas as pd
import os
import numpy as np
import re



data = pd.read_csv("flavors_of_cacao.csv")
column_names=['Brand Name', 'Specific Bar Origin','REF', 'Review Date', 'Cocoa Percent', 'Company Location', 'Rating','Bean Type', 'Bean Origin']
data=data.rename(columns=dict(zip(data.columns,column_names)))

# Bean Origin-------------
# drop null space
t=data["Bean Origin"].notnull()
data=data[t]
print(data.isnull().sum())

q=data["Bean Type"].notnull()
data=data[q]
print(data.isnull().sum())
#drop space which showed ? in the csv
filter_0 = data["Bean Origin"]!='\xa0'
data=data[filter_0]
# typo
data["Bean Origin"]=data["Bean Origin"].replace("Domincan Republic","Dominican Republic")
data["Bean Origin"]=data["Bean Origin"].replace("Trinidad-Tobago","Trinidad")
data["Specific Bar Origin"]=data["Specific Bar Origin"].replace("Trinidad-Tobago","Trinidad")
data["Bean Origin"]=data["Bean Origin"].replace("Sao Tome & Principe","Sao Tome")
data["Specific Bar Origin"]=data["Specific Bar Origin"].replace("Sao Tome & Principe","Sao Tome")
#align the expression
data["Bean Origin"]=data["Bean Origin"].replace("Venezuela/ Ghana","Venezuela, Ghana")
#get out of () to simplify the data

for i in data.index:
    sub_str=str(data.loc[i,"Bean Origin"])
    sub_str=sub_str.split(" (")
    data.loc[i,"Bean Origin"]=sub_str[0].strip()

# Brand Name-------------
#get out of () to simplify the data
for i in data.index:
    sub_str = str(data.loc[i,"Brand Name"])
    sub_str = sub_str.split(" (")
    data.loc[i,"Brand Name"] = sub_str[0].strip()

#Andrews code for Bean Origin and Bean Type----------------------------------
## Text preparation (correction) func
def txt_prep(text):
    replacements = [
        ['-', ', '], ['/ ', ', '], ['/', ', '], ['\(', ', '], [' and', ', '], [' &', ', '], ['\)', ''],
        ['Dom Rep|DR|Domin Rep|Dominican Rep,|Domincan Republic', 'Dominican Republic'],
        ['Mad,|Mad$', 'Madagascar, '],
        ['PNG', 'Papua New Guinea, '],
        ['Guat,|Guat$', 'Guatemala, '],
        ['Ven,|Ven$|Venez,|Venez$', 'Venezuela, '],
        ['Ecu,|Ecu$|Ecuad,|Ecuad$', 'Ecuador, '],
        ['Nic,|Nic$', 'Nicaragua, '],
        ['Cost Rica', 'Costa Rica'],
        ['Mex,|Mex$', 'Mexico, '],
        ['Jam,|Jam$', 'Jamaica, '],
        ['Haw,|Haw$', 'Hawaii, '],
        ['Gre,|Gre$', 'Grenada, '],
        ['Tri,|Tri$', 'Trinidad, '],
        ['C Am', 'Central America'],
        ['S America', 'South America'],
        [', $', ''], [',  ', ', '], [', ,', ', '], ['\xa0', ' '],[',\s+', ','],
        [' Bali', ',Bali']
    ]
    for i, j in replacements:
        text = re.sub(i, j, text)
    return text

data['Bean Origin'].str.replace('.', '').apply(txt_prep).unique()

## Replace brand name
##data['Brand Name'] = data['Brand Name'].fillna(data['Brand Name'])
##data['Brand Name'].isnull().value_counts()
## Text preparation (correction) func
def txt_prep1(text):
    replacements = [
        ['Artisan du Chocolat \(Casa Luker\)','Artisan du Chocolat'],
        ['Aequare \(Gianduja\)','Aequare'],['Akesson\'s \(Pralus\)','Akesson'],
        ['Amatller \(Simon Coll\)','Amatller'],['Beschle \(Felchlin\)','Beschle'],
        ['Black River \(A Morin\)','Black River'],['Bouga Cacao \(Tulicorp\)','Bouga Cacao'],['Cacaosuyo \(Theobroma Inversiones\)','Cacaosuyo'],
        ['Cacaoyere \(Ecuatoriana\)','Cacaoyere'],['Caoni \(Tulicorp\)','Caoni'],['Chchukululu \(Tulicorp\)','Chchukululu'],
        ['Chokolat Elot \(Girard\)','Chokolat Elot'],['Christopher Morel \(Felchlin\)','Christopher Morel'],
        ['Chuao Chocolatier \(Pralus\)','Chuao Chocolatier'],['Compania de Chocolate \(Salgado\)','Compania de Chocolate'],
        ['Cote d\' Or \(Kraft\)','Cote d'],['Dean and Deluca \(Belcolade\)','Dean and Deluca'],
        ['Debauve & Gallais \(Michel Cluizel\)','Debauve & Gallais'],['Dole \(Guittard\)','Dole'],
        ['Dolfin \(Belcolade\)','Dolfin'],['Eclat \(Felchlin\)','Eclat'],['Enric Rovira \(Claudio Corallo\)','Enric Rovira'],
        ['Erithaj \(A Morin\)','Erithaj'],['Ethel\'s Artisan \(Mars\)','Ethel'],['Fearless \(AMMA\)','Fearless'],
        ['Forteza \(Cortes\)','Forteza'],['Friis Holm \(Bonnat\)','Friis Holm'],['Green & Black\'s \(ICAM\)','Green & Black'],
        ['Heirloom Cacao Preservation \(Brasstown\)','Heirloom Cacao Preservation'],
        ['Heirloom Cacao Preservation \(Fruition\)','Heirloom Cacao Preservation'],
        ['Heirloom Cacao Preservation \(Guittard\)','Heirloom Cacao Preservation'],
        ['Heirloom Cacao Preservation \(Manoa\)','Heirloom Cacao Preservation'],
        ['Heirloom Cacao Preservation \(Millcreek\)','Heirloom Cacao Preservation'],
        ['Heirloom Cacao Preservation \(Mindo\)','Heirloom Cacao Preservation'],
        ['Heirloom Cacao Preservation \(Zokoko\)','Heirloom Cacao Preservation'],
        ['Hoja Verde \(Tulicorp\)','Hoja Verde'],['Hotel Chocolat \(Coppeneur\)','Hotel Chocolat'],['Idilio \(Felchlin\)','Idilio'],
        ['Kallari \(Ecuatoriana\)','Kallari'],['Kaoka \(Cemoi\)','Kaoka'],['LA Burdick \(Felchlin\)','LA Burdick'],
        ['La Maison du Chocolat \(Valrhona\)','La Maison du Chocolat'],['Lake Champlain \(Callebaut\)','Lake Champlain'],
        ['Madecasse \(Cinagra\)','Madecasse'],['Malagasy \(Chocolaterie Robert\)','Malagasy'],['Malie Kai \(Guittard\)','Malie Kai'],
        ['Menakao \(aka Cinagra\)','Menakao'],['Muchomas \(Mesocacao\)','Muchomas'],['Na�ve','Naive'],['Neuhaus \(Callebaut\)','Neuhaus'],
        ['Oialla by Bojessen \(Malmo\)','Oialla by Bojessen'],['Original Beans \(Felchlin\)','Original Beans'],
        ['Pomm \(aka Dead Dog\)','Pomm'],['Quetzalli \(Wolter\)','Quetzalli'],['Republica del Cacao \(aka Confecta\)','Republica del Cacao'],
        ['Robert \(aka Chocolaterie Robert\)','Robert'],['Rococo \(Grenada Chocolate Co\)','Rococo'],['Santander \(Compania Nacional\)','Santander'],
        ['Sprungli \(Felchlin\)','Sprungli'],['Stella \(aka Bernrain\)','Stella'],['Tablette \(aka Vanillabeans\)','Tablette'],
        ['To\'ak \(Ecuatoriana\)','To\'ak'],['Tobago Estate \(Pralus\)','Tobago Estate'],['Tsara \(Cinagra\)','Tsara'],
        ['Vanleer \(Barry Callebaut\)','Vanleer'],['Vao Vao \(Chocolaterie Robert\)','Vao Vao'],['Vietcacao \(A Morin\)','Vietcacao'],['Vintage Plantations \(Tulicorp\)','Vintage Plantations'],
    ]
    for i, j in replacements:
        text = re.sub(i, j, text)
    return text
data['Brand Name'].str.replace('.', '').apply(txt_prep1).unique()
## Replace brand name
data['Brand Name'] = data['Brand Name'].fillna(data['Brand Name'])
data['Brand Name'].isnull().value_counts()
#Specific Bar Origin -----------------------------------------------------------------------------
# re-organize this item , this item should be bar's name usually named after bean's country' and county
#country -> country-> special name
for i in data.index:
    if data.loc[i, "Bean Origin"] != data.loc[i, "Specific Bar Origin"]:
        sub_str=data.loc[i,"Specific Bar Origin"].split(",")
        if sub_str[0]==data.loc[i , "Bean Origin"]:
            data.loc[i,"Specific Bar Origin"]=sub_str[1]
        else:
            data.loc[i,"Specific Bar Origin"]=sub_str[0]

# Bean Type-----------
## simplifed data
replacements = [
    ['Forastero (Arriba) ASS','Forastero (Arriba)'],
    ['Forastero (Arriba) ASSS','Forastero (Arriba)'],
    ['Forastero (Nacional)','Nacional'],
    ['Amazon mix','Amazon'],
    ['Amazon, ICS','Amazon'],
    ['Blend-Forastero,Criollo','Trinitario, Criollo'],
    ['Criollo, +','Criollo'],
    ['Trinitario (85% Criollo)','Trinitario, Criollo'],
    ['Forastero(Arriba, CCN)','Forastero (Arriba)'],
    ]
for i in range(len(replacements)):
    x=replacements[i][0]
    y=replacements[i][1]
    data["Bean Type"] = data["Bean Type"].replace(x,y)

data.to_csv("data_clean.csv")

# fill in missing data by Kmean
data=data.replace(np.NaN,'\xa0')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# decide bean type by bean origin/brand name/ specific bar origin
data_BT=data[["Brand Name","Specific Bar Origin","Bean Type","Bean Origin"]]

print(data_BT.isnull().sum())
Seq=pd.DataFrame({"sequence":range(len(data.index)),"Index":data.index})
Seq=Seq.set_index("Index")

#creat filter  for Bean Type
filter_BT=data["Bean Type"]!='\xa0'
filter_BT_application=data["Bean Type"]=='\xa0'
#create dummy X
X_data_dummy=pd.get_dummies(data_BT[["Brand Name","Specific Bar Origin","Bean Origin"]])
#sepertate data into train and application
X_data_dummy_train = X_data_dummy[filter_BT]
X=X_data_dummy_train.values
Seq_app = Seq[filter_BT_application]
Y_data = data_BT["Bean Type"].astype(str)
Y_data_train = Y_data[filter_BT]

#fit function
class_le = LabelEncoder()
y = class_le.fit_transform(Y_data_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

clf=SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test,y_pred))
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

#our result
X_data_dummy_app = X_data_dummy[filter_BT_application]
X_app = X_data_dummy_app.values
y_re = clf.predict(X_app)
y_re_2=class_le.inverse_transform(y_re)

#matrix
conf_matrix = confusion_matrix(y_test, y_pred)
t=np.unique(y_test)
y_col = class_le.inverse_transform(t)
class_names =y_col.tolist()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels())

#Non-liner SVM
svm3 = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm3.fit(X_train, y_train)
y_pred_2 = svm3.predict(X_test)

print(classification_report(y_test,y_pred_2))
print("Accuracy : ", accuracy_score(y_test, y_pred_2) * 100)

y_re_3=svm3.predict(X_app)
y_re_3=class_le.inverse_transform(y_re_3)


#insert y_re_3 into those unknown data
ref=pd.DataFrame({"prediction": y_re_2,"Index": Seq_app.index,"sequence": range(len(y_re_3))})

for i in ref["sequence"]:
    x_index=ref.loc[i,"Index"]
    xo=ref.loc[i,"prediction"]
    if data.loc[x_index, "Bean Type"]=='\xa0':
        data.loc[x_index, "Bean Type"] = xo

data.to_csv("data clean with Bean Type.csv")