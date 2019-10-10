import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
labelEncoder =  preprocessing.LabelEncoder()
#reading the data from the sourse 
df = pd.read_csv("https://web.stanford.edu/~hastie/ElemStatLearn//datasets/SAheart.data")

#print(df)
column = ['names','sbp','tobacco','ldl','adiposity','famhist','type','obesity','alcohol','age','chd']

df.columns = column

print(df.head())

print(df.describe())

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['famhist']=encoder.fit_transform(df['famhist'])
df['chd']=encoder.fit_transform(df['chd'])

print(df.head(5))

from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range =(0,100))


# setting scale of max min value for sbp in range of 0-100, normalise
df['sbp'] = scale.fit_transform(df['sbp'].values.reshape(-1,1))

print(df.head())

print(df.describe())

df.head(50).plot(kind='area',figsize=(10,5))
plt.show()




#Distribution of Obesity according to the age

df.plot(x='age',y='obesity',kind='scatter',figsize =(10,5))
plt.show()

# splitting the data into test and train  having a test size of 20% and 80% train size
from sklearn.model_selection import train_test_split
col = ['sbp','tobacco','ldl','adiposity','famhist','type','obesity','alcohol','age']
X_train, X_test, y_train, y_test = train_test_split(df[col], df['chd'], test_size=0.2, random_state=1234)


#HeatMap of relation features 
sns.set()
sns.heatmap(X_train.head(10),robust = True)
plt.show()

X_all = df[col]
y_all = df['chd']

from sklearn import svm
svm_clf = svm.SVC(kernel ='linear')

svm_clf.fit(X_train,y_train)

y_pred_svm =svm_clf.predict(X_test)


print(y_pred_svm)

# Let's create the confusion matrix
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(cm_svm)

from sklearn.metrics import accuracy_score
svm_result = accuracy_score(y_test,y_pred_svm)
print("Accuracy :",svm_result)

recall_svm = cm_svm[0][0]/(cm_svm[0][0] + cm_svm[0][1])
precision_svm = cm_svm[0][0]/(cm_svm[0][0]+cm_svm[1][1])
print(recall_svm,precision_svm)

# the results of this section is Accuracy : 0.741 Recall : 0.85 Precision : 0.739


# Now we are do the test for KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors =5,n_jobs = -1,leaf_size = 60,algorithm='brute')



knn_clf.fit(X_train,y_train)


KNeighborsClassifier(algorithm='brute', leaf_size=60, metric='minkowski',
           metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
           weights='uniform')



y_pred_knn = knn_clf.predict(X_test)
print(y_pred_knn)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)

knn_result = accuracy_score(y_test,y_pred_knn)
print(knn_result)

recall_knn = cm_knn[0][0]/(cm_knn[0][0] + cm_knn[0][1])
precision_knn = cm_knn[0][0]/(cm_knn[0][0]+cm_knn[1][1])
print(recall_knn,precision_knn)


#The results of KNN is => Accuracy : 0.645 Recall : 0.816 Precision : 0.816


# The last test is Kfold cross validation  and we have 10 folds

from sklearn.model_selection import KFold

fold_list =[]
accuracy_list =[]
"""def run_kfold(knn_clf):
    kf = KFold(297, n_splits=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        knn_clf.fit(X_train, y_train)
        predictions = knn_clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
        fold_list.append(fold)
        accuracy_list.append(accuracy)
        mean_outcome = np.mean(outcomes)
        print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(knn_clf)"""


#ANN Multilayer Perceptron Classifier

#Input nodes : 9
#Hidden layers :2
#Each hidden Layer Hold 14 neuron
#Output layer : 2

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


ann_clf = MLPClassifier()

#Parameters
parameters = {'solver': ['lbfgs'],
             'alpha':[1e-4],
             'hidden_layer_sizes':(9,14,14,2),   # 9 input, 14-14 neuron in 2 layers,1 output layer
             'random_state': [1]}


# Type of scoring to compare parameter combos 
acc_scorer = make_scorer(accuracy_score)

# Run grid search 
grid_obj = GridSearchCV(ann_clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Pick the best combination of parameters
ann_clf = grid_obj.best_estimator_



# Fit the best algorithm to the data 
ann_clf.fit(X_train, y_train)



y_pred_ann = ann_clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ann = confusion_matrix(y_test, y_pred_ann)
print(cm_ann)

ann_result = accuracy_score(y_test,y_pred_ann)
print(ann_result)

recall_ann = cm_ann[0][0]/(cm_ann[0][0] + cm_ann[0][1])
precision_ann = cm_ann[0][0]/(cm_ann[0][0]+cm_ann[1][1])
print(recall_ann,precision_ann)



#Accuracy : 0.763 Recall : 0.866 Precision : 0.732



#Sequential Model Input node :9 Hidden layer :2 Each layer hold 6 neuron

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 9))

# Adding the second hidden layer
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_annk = confusion_matrix(y_test, y_pred)


print(cm_annk)

recall_annk = cm_annk[0][0]/(cm_annk[0][0] + cm_annk[0][1])
precision_annk = cm_annk[0][0]/(cm_annk[0][0]+cm_annk[1][1])
print(recall_annk,precision_annk)

results ={'Accuracy': [svm_result*100,knn_result*100,ann_result*100],
          'Recall': [recall_svm*100,recall_knn*100,recall_ann*100],
          'Precision': [precision_svm*100,precision_knn*100,precision_ann*100]}
index = ['SVM','KNN','ANN']

results =pd.DataFrame(results,index=index)

fig =results.plot(kind='bar',title='Comaprison of models',figsize =(9,9)).get_figure()
fig.savefig('Final Result.png')

