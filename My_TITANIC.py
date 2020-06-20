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
# Algorithms

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#4. Modélisation avec Random Forest
#Pour cette modélisation nous nous servirons de Title, Embarked, Sex, Child, Fare, Pclass, FsizeD. Nous excluons plusieurs variables : Cabin, où il y'a trop de valeurs manquantes, ainsi que Age, SibSp, et Parch, qui ne contribuent pas à l'amélioration du pouvoir prédictif du modèle (j'ai essayés plusieurs combinaisons). PassengerID, Ticket et Name sont des valeurs uniques par passagers, nous les excluons aussi.
#Random Forest est un algorithme de classification "simple à utiliser", nous ne sommes pas obligés de normaliser les données, ni de valider le modèle par cross-validation, par exemple.

#regression linéaire pour l'age
data['familymembers']=data.SibSp+data.Parch
datanotna=data[data['Age'] .notna()]
y=datanotna['Age']
x=datanotna[['Survived','Pclass','Sex','Embarked','familymembers']]
from sklearn.model_selection import train_test_split
x_train,x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2)
print('Train set',x_train.shape)
print('validation',x_test.shape)

#determiner le best estimators
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()
ne = np.arange(1,20)
param_grid = {'n_estimators' : ne}
from sklearn.model_selection import GridSearchCV
rf_cv = GridSearchCV(random_forest, param_grid=param_grid, cv=5)
rf_cv.fit(x_train, y_train)
print('Best value of n_estimators:',rf_cv.best_params_)
print('Best score:',rf_cv.best_score_*100)


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 45, random_state = 45)
rf.fit(x_train, y_train);
print('train score:',rf.score(x_train,y_train))
print('validation score:',rf.score(x_test,y_test))



#graphe
from sklearn.model_selection import validation_curve
k=np.arange(1,100)
train_score,val_score=validation_curve(rf,x_train,y_train,'n_estimators',k,cv=5)
plt.plot(k,val_score.mean(axis=1),label='validation')
plt.plot(k,train_score.mean(axis=1),label='train')
plt.ylabel('score')
plt.xlabel('n_estimators')
plt.legend()

#new data
dataisna=data[data['Age'] .isna()]
y_final=dataisna['Age']
x_final=dataisna[['Survived','Pclass','Sex','familymembers','Embarked']]
y_final=rf.predict(x_final)
dataisna['Age']=y_final
data_sklearn=pd.concat([dataisna,datanotna])
print(data_sklearn)


#♥survived
X = data_sklearn.drop("Survived", axis=1)
Y = data_sklearn["Survived"]
X.shape, Y.shape, data_sklearn.shape
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

print(X)

#Random Forest:

random_forest = RandomForestClassifier(n_estimators=13)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(x_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)
#avec validation croiseé
random_forest = RandomForestClassifier()
ne = np.arange(1,20)
param_grid = {'n_estimators' : ne}
from sklearn.model_selection import GridSearchCV
rf_cv = GridSearchCV(random_forest, param_grid=param_grid, cv=5)
rf_cv.fit(X_train, Y_train)
print('Best value of n_estimators:',rf_cv.best_params_)
print('Best score:',rf_cv.best_score_*100)




importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(8)
importances.plot.bar()


#K Nearest Neighbor:

knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train)  
Y_pred = knn.predict(x_test)  
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)
#avec validation croisée
from sklearn.model_selection import cross_val_score
KNN =  KNeighborsClassifier()
scores = cross_val_score(KNN, X, Y, cv=10)
scores.sort()
accuracy = scores.mean()
print(scores)
print(accuracy)

#Linear Support Vector Machine:
svc=SVC()
svc.fit(X_train, Y_train)
Y_pred2 = svc.predict(x_test)
acc_svc = round(svc.score(x_test, y_test) * 100, 2)

print(acc_svc)

#avec validation croisée

from sklearn.model_selection import cross_val_score
SV =  SVC()
scores = cross_val_score(SV, X, Y, cv=10)
scores.sort()
accuracy = scores.mean()
print(scores)
print(accuracy)


#Which is the best Model ?
results = pd.DataFrame({
    'Model': ['Random Forest', 'KNN', 'SVM'],
    'Score': [ acc_random_forest,acc_knn,acc_svc ]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(3)

objects = ( 'KNN','Random Forest', 'SVM')
x_pos = np.arange(len(objects))
accuracies1 = [acc_knn,acc_random_forest ,acc_svc]
plt.bar(x_pos, accuracies1, align='center', alpha=0.5, color='yellow')
plt.xticks(x_pos, objects, rotation='vertical')
plt.ylabel('Accuracy')
plt.title('Classifier Outcome')
plt.show()

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
#        3éme phase
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------


data3=pd.read_csv(r"C:\Users\user\Desktop\projet machine learning_Titanic/test.csv")
print(data3.shape)#taille de fichier
data3.head()  #affichier les premiers lignes
print(data3)#affichier le fchier
print(data3.describe)
data3.columns#afficher les noms des colonnes de ce fichier
data3.dtypes.value_counts() #le nombre de type des variables
#missig data
print(data3.isna().sum())
print(100 * data3.isnull().sum() / len(data3))

#drop missig data "cabin"
data3=data3.drop(['Cabin'],axis=1)
#drop missig data "Fare"
data3=data3.drop(['Fare'],axis=1)


#supprimer les trois colonnes 
data3=data3.drop(['Name'],axis=1)
data3=data3.drop(['Ticket'],axis=1)
data3=data3.drop(['PassengerId'],axis=1)

# Convert the Embarked classes to integer form
data3["Embarked"][data3["Embarked"] == "S"] = 0
data3["Embarked"][data3["Embarked"] == "C"] = 1
data3["Embarked"][data3["Embarked"] == "Q"] = 2
print(data3["Embarked"].head(28))

# Convert the sex classes to integer form
data3["Sex"][data3["Sex"] == "male"] = 0
data3["Sex"][data3["Sex"] == "female"] = 1
print(data3["Sex"].head(28))

print(data3)

from sklearn.ensemble import RandomForestClassifier


data3['familymembers']=data3.SibSp+data3.Parch
datanotna=data3[data3['Age'] .notna()]
y=datanotna['Age']
x=datanotna[['Pclass','Sex','Embarked','familymembers']]
from sklearn.model_selection import train_test_split
x_train,x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2)
print('Train set',x_train.shape)
print('Test set',x_test.shape)


#determiner le best estimators
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()
ne = np.arange(1,20)
param_grid = {'n_estimators' : ne}
from sklearn.model_selection import GridSearchCV
rf_cv = GridSearchCV(random_forest, param_grid=param_grid, cv=5)
rf_cv.fit(x_train, y_train)
print('Best value of n_estimators:',rf_cv.best_params_)
print('Best score:',rf_cv.best_score_*100)




# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 19, random_state = 45)
rf.fit(x_train, y_train);
print('train score:',rf.score(x_train,y_train))
print('test score:',rf.score(x_test,y_test))


#graphe
from sklearn.model_selection import validation_curve
k=np.arange(1,100)
train_score,val_score=validation_curve(rf,x_train,y_train,'n_estimators',k,cv=5)
plt.plot(k,val_score.mean(axis=1),label='validation')
plt.plot(k,train_score.mean(axis=1),label='train')
plt.ylabel('score')
plt.xlabel('n_estimators')
plt.legend()

#new data
dataisna=data3[data3['Age'] .isna()]
y_final=dataisna['Age']
x_final=dataisna[['Pclass','Sex','familymembers','Embarked']]
y_final=rf.predict(x_final)
dataisna['Age']=y_final
data_sklearn=pd.concat([dataisna,datanotna])
print(data_sklearn)



#Predicting
clf = RandomForestClassifier(ne)
clf.fit(X_train,data["Survived"].values)
predicted_Y = clf.predict(x_test )


#Creating a dataframe with PassengerIds and the predicted variable Survived
passengerId = np.array(data_sklearn['PassengerId']).astype(int)
submission = pd.DataFrame({ 'PassengerId' : passengerId, 'Survived' : predicted_Y })

submission.to_csv('submission4.csv', index=False)




















