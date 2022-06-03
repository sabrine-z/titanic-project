# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:25:03 2020

@author: sabrine
"""

#----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#        1ère phase
#---------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


data=pd.read_csv(r"C:\Users\user\Desktop\projet machine learning_Titanic/train.csv")
print(data.shape)#taille de fichier
data.head()  #affichier les premiers lignes
print(data)#affichier le fchier
print(data.describe)
data.columns#afficher les noms des colonnes de ce fichier
data.dtypes.value_counts() #le nombre de type des variables

data['Pclass'].value_counts()
data['Sex'].value_counts()
data['SibSp'].value_counts()
data['Parch'].value_counts()
data['Ticket'].value_counts()
data['Cabin'].value_counts()
data['Embarked'].value_counts()
data['Fare'].value_counts()



#number of survived 

sns.set(rc={'figure.figsize':(15,8)})
_ = sns.countplot(x="Survived", data=data)
print('Total Survivor :',data['Survived'].sum())
print('Total  of female survived:',data[data.Sex == 'female'].Survived.sum())
print('Total  of female passangers:',data[data.Sex == 'female'].Survived.count())
print('Percentage of female survived',data[data.Sex == 'female'].Survived.sum()/data[data.Sex == 'female'].Survived.count())
print('Total  of male survived:',data[data.Sex == 'male'].Survived.sum())
print('Total  of male passangers:',data[data.Sex == 'male'].Survived.count())
print('Percentage of male survived:',data[data.Sex == 'male'].Survived.sum()/data[data.Sex == 'male'].Survived.count())

data.Pclass.describe()
data.Fare.describe()
data.Sex.describe()
data.Age.describe()




###########Visualization#######################

# Countplots for all categorical columns
cols = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
nr_rows = 2
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        
        i = r*nr_cols+c       
        ax = axs[r][c]
        sns.countplot(data[cols[i]], hue=data["Survived"], ax=ax)
        ax.set_title(cols[i], fontsize=14, fontweight='bold')
        ax.legend(title="survived", loc='upper center') 
        
plt.tight_layout()   


sns.barplot(x='Pclass',y='Survived',data=data)
data[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)



sns.barplot(x='Parch',y='Survived',data=data)
data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Comparing the Embarked feature against Survived
sns.barplot(x='Embarked',y='Survived',data=data)
data[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Distribution of Age as function of Pclass, Sex and Survived
bins = np.arange(0, 80, 5)
g = sns.FacetGrid(data, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sns.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()  
plt.show()


#Disribution of Fare as function of Pclass, Sex and Survived
bins = np.arange(0, 550, 50)
g = sns.FacetGrid(data, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)
g.map(sns.distplot, 'Fare', kde=False, bins=bins, hist_kws=dict(alpha=0.6))
g.add_legend()  
plt.show()  

#Disribution of Pclass as function of Embarked, Sex and Survived
bins = np.arange(0, 3, 1)
g= sns.FacetGrid(data, row='Embarked', size=2.2, aspect=1.6)
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
g.add_legend()
plt.show()

#missing data
print(data.isna().sum())
print(100 * data.isnull().sum() / len(data))

#2éme méthode plus présise
total = data.isnull().sum().sort_values(ascending=False)
percent_1 = data.isnull().sum()/data.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)
#graphe
fig, ax = plt.subplots(figsize=(9,5))
sns.heatmap(data.isnull(), cbar=False, cmap="YlGnBu_r")
plt.show()


#dealing with missing data
datanotna=data[data["Age"].notna()]
dataisna=data[data["Age"].isna()]
datanotnaf=datanotna[datanotna.Sex == 'female']
datanotnam=datanotna[datanotna.Sex == 'male']
dataisnaf=dataisna[dataisna.Sex == 'female']
dataisnam=dataisna[dataisna.Sex == 'male']
print(datanotna)#714
print(dataisna)#177
print(datanotnaf)#261 femmes
print(datanotnam)#453 hommes
print(dataisnaf)#53 femmes
print(dataisnam)#124 hommes
 
sns.factorplot('Pclass',data=dataisna,hue='Sex',kind='count')#on ne peut pas eliminer sex et pclass 

sns.factorplot('Parch',data=dataisna,hue='Survived',kind='count')#on peut  eliminer  parch =3,4,5,6


sns.factorplot('Embarked',data=dataisna,hue='Survived',kind='count')#on ne peut pas coclure


#parch=0
print(dataisna[dataisna["Parch"]==0])#157 personne
print(datanotna[datanotna["Parch"]==0])#521 personnes
parch0=datanotna[datanotna["Parch"]==0]
x0=parch0['Age'].mean()
print(x0)
parchfinal0=dataisna[dataisna["Parch"]==0]
parchfinal0['Age']=x0
print(parchfinal0['Age'])
#femme:
print(dataisnaf[dataisnaf["Parch"]==0])#41
print(datanotnaf[datanotnaf["Parch"]==0])#153
parch0f=datanotnaf[datanotnaf["Parch"]==0]
x0f=parch0f['Age'].mean()
print(x0f)
parchfinal0f=dataisnaf[dataisnaf["Parch"]==0]
parchfinal0f['Age']=x0f
print(parchfinal0f['Age'])
#homme:
print(dataisnam[dataisnam["Parch"]==0])#116
print(datanotnam[datanotnam["Parch"]==0])#368
parch0m=datanotnam[datanotnam["Parch"]==0]
x0m=parch0m['Age'].mean()
print(x0m)
parchfinal0m=dataisnam[dataisnam["Parch"]==0]
parchfinal0m['Age']=x0m
print(parchfinal0m['Age'])

parchfinal0=pd.concat([parchfinal0f,parchfinal0m])
print(parchfinal0)


#parch=1
print(dataisna[dataisna["Parch"]==1])#8 personnes
print(datanotna[datanotna["Parch"]==1])#110 personnes
parch1=datanotna[datanotna["Parch"]==1]
x1=parch1['Age'].mean()
print(x1)
parchfinal1=dataisna[dataisna["Parch"]==1]
parchfinal1['Age']=x1
print(parchfinal1['Age'])
#femme:
print(dataisnaf[dataisnaf["Parch"]==1])#5
print(datanotnaf[datanotnaf["Parch"]==1])#55
parch1f=datanotnaf[datanotnaf["Parch"]==1]
x1f=parch1f['Age'].mean()
print(x1f)
parchfinal1f=dataisnaf[dataisnaf["Parch"]==1]
parchfinal1f['Age']=x1f
print(parchfinal1f['Age'])
#homme:
print(dataisnam[dataisnam["Parch"]==1])#3
print(datanotnam[datanotnam["Parch"]==1])#55
parch1m=datanotnam[datanotnam["Parch"]==1]
x1m=parch1m['Age'].mean()
print(x1m)
parchfinal1m=dataisnam[dataisnam["Parch"]==1]
parchfinal1m['Age']=x1m
print(parchfinal1m['Age'])


parchfinal1=pd.concat([parchfinal1f,parchfinal1m])
print(parchfinal1)

#parch=2
print(dataisna[dataisna["Parch"]==2])#12 personnes
print(datanotna[datanotna["Parch"]==2])#68 personnes
parch2=datanotna[datanotna["Parch"]==2]
x2=parch2['Age'].mean()
print(x2)
parchfinal2=dataisna[dataisna["Parch"]==2]
parchfinal2['Age']=x2
print(parchfinal2['Age'])
#femme:
print(dataisnaf[dataisnaf["Parch"]==2])#7
print(datanotnaf[datanotnaf["Parch"]==2])#42
parch2f=datanotnaf[datanotnaf["Parch"]==2]
x2f=parch2f['Age'].mean()
print(x2f)
parchfinal2f=dataisnaf[dataisnaf["Parch"]==2]
parchfinal2f['Age']=x2f
print(parchfinal2f['Age'])
#homme:
print(dataisnam[dataisnam["Parch"]==2])#5
print(datanotnam[datanotnam["Parch"]==2])#26
parch2m=datanotnam[datanotnam["Parch"]==2]
x2m=parch2m['Age'].mean()
print(x2m)
parchfinal2m=dataisnam[dataisnam["Parch"]==2]
parchfinal2m['Age']=x2m
print(parchfinal2m['Age'])

parchfinal2=pd.concat([parchfinal2f,parchfinal2m])
print(parchfinal2)


#new data
data=pd.concat([parchfinal0,parchfinal1,parchfinal2,datanotna])
print(data)

#drop missig data "cabin"
data=data.drop(['Cabin'],axis=1)

data['Embarked'].describe()

# Fill NAN with the most frequent value:
data["Embarked"] = data["Embarked"].fillna("S")


#resultat
print(data.isna().sum())
print(data)

#supprimer les trois colonnes 
data=data.drop(['Name'],axis=1)
data=data.drop(['Ticket'],axis=1)
data=data.drop(['PassengerId'],axis=1)

print(data)

# Convert the Embarked classes to integer form
data["Embarked"][data["Embarked"] == "S"] = 0
data["Embarked"][data["Embarked"] == "C"] = 1
data["Embarked"][data["Embarked"] == "Q"] = 2
print(data["Embarked"].head(28))

# Convert the sex classes to integer form
data["Sex"][data["Sex"] == "male"] = 0
data["Sex"][data["Sex"] == "female"] = 1
print(data["Sex"].head(28))

#resultat finale
print(data)
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#        2éme phase
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

#regression linéaire
data['familymembers']=data.SibSp+data.Parch

datanotna=data[data['Age'] .notna()]
y=datanotna['Age']
x=datanotna[['Survived','Pclass','Sex','Embarked','familymembers']]
from sklearn.model_selection import train_test_split
x_train,x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2)
print('Train set',x_train.shape)
print('validation',x_test.shape)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 45, random_state = 45)

# Train the model on training data
rf.fit(x_train, y_train);
print('train score:',rf.score(x_train,y_train))
print('test score:',rf.score(x_test,y_test))
from sklearn.model_selection import cross_val_score
cross_val_score(RandomForestRegressor(),x_train,y_train,cv=5)
