from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

################################Preparing Y_train
Y_train = train.pop('Survived')

#checking for missing values
Y_train.isna().sum()  #0 found

Y_train = Y_train.values.reshape(train.shape[0], 1)

#np.save('npy_files/Y_train.npy', Y_train)
#Y_train = np.load('npy_files/Y_train.npy')

###############################Joining Dataset
dataset = pd.concat((train, test), axis = 0)
#del train, test

#Missing values
missing = dataset.isna().sum()

##############################PassengerId
dataset.pop('PassengerId')

##############################Cabin
dataset.pop('Cabin')

##############################Ticket
dataset.pop('Ticket')


##############################Pclass
Pclass = dataset.pop('Pclass')

#checking for missing values
Pclass.isna().sum()  #0 found

Pclass = Pclass.values.reshape(dataset.shape[0], 1)

#np.save('npy_files/Pclass.npy', Pclass)
#Pclass = np.load('npy_files/Pclass.npy')

##############################Sex
Sex = dataset.pop('Sex')

#checking for missing values
Sex.isna().sum()  #0 found

Sex = Sex.values.reshape(dataset.shape[0], 1)
Sex = LabelEncoder().fit_transform(Sex).reshape(dataset.shape[0], 1)

#np.save('npy_files/Sex.npy', Sex)
#Sex = np.load('npy_files/Sex.npy')

##############################SibSp
SibSp = dataset.pop('SibSp')

#checking for missing values
SibSp.isna().sum()  #0 found

SibSp = SibSp.values.reshape(dataset.shape[0], 1)

#np.save('npy_files/SibSp.npy', SibSp)
#SibSp = np.load('npy_files/SibSp.npy')

##############################Parch
Parch = dataset.pop('Parch')

#checking for missing values
Parch.isna().sum()  #0 found

Parch = Parch.values.reshape(dataset.shape[0], 1)

#np.save('npy_files/Parch.npy', Parch)
#Parch = np.load('npy_files/Parch.npy')

##############################Fare
Fare = dataset.pop('Fare')

#checking for missing values
Fare.isna().sum()  #1 found

#Filling missing value with median value of data
Fare.fillna(Fare.median(), inplace = True)

#checking for missing values
Fare.isna().sum()  #0 found

Fare = Fare.values.reshape(dataset.shape[0], 1)

Fare = StandardScaler().fit_transform(Fare).reshape(dataset.shape[0], 1)

#np.save('npy_files/Fare.npy', Fare)
#Fare = np.load('npy_files/Fare.npy')

###############################Embarked
Embarked = dataset.pop('Embarked')

#checking for missing values
Embarked.isna().sum()  #1 found

#Filling missing value with median value of data
Embarked.fillna(Embarked.mode()[0], inplace = True)

#checking for missing values
Embarked.isna().sum()  #0 found

Embarked = Embarked.values.reshape(dataset.shape[0], 1)
Embarked = LabelEncoder().fit_transform(Embarked).reshape(dataset.shape[0], 1)
Embarked = OneHotEncoder().fit_transform(Embarked).toarray()[:, 1:]

#np.save('npy_files/Embarked.npy', Embarked)
#Embarked = np.load('npy_files/Embarked.npy')

###############################Age
Age = dataset.pop('Age')

#checking for missing values
Age.isna().sum()  #263 found

#Filling missing value with median value of data
Age.fillna(Age.mean(), inplace = True)    #Try Median???

#checking for missing values
Age.isna().sum()  #0 found

Age = Age.values.reshape(dataset.shape[0], 1)
Age = StandardScaler().fit_transform(Age).reshape(dataset.shape[0], 1)

#np.save('npy_files/Age.npy', Age)
#Age = np.load('npy_files/Age.npy')

##############################FamilySize -> Feature Engineering from SibSp and Parch
FamilySize = np.add(SibSp, Parch)
FamilySize = np.add(FamilySize, np.ones((dataset.shape[0], 1)))

#Apply Standard Scaler??
#FamilySize = StandardScaler().fit_transform(FamilySize).reshape(dataset.shape[0], 1)

#np.save('npy_files/FamilySize.npy', FamilySize)
#FamilySize = np.load('npy_files/FamilySize.npy')

##############################IsAlone -> Feature Engineering from SibSp and Parch
IsAlone = [0 if i[0]>1 else 1 for i in FamilySize]
IsAlone = np.array(IsAlone).reshape((dataset.shape[0], 1))

#np.save('npy_files/IsAlone.npy', IsAlone)
#IsAlone = np.load('npy_files/IsAlone.npy')

##############################Name
Name = dataset['Name']

Name = Name.apply(lambda x: x.split(', ')[1].split('. ')[0])

#Checking for missing Values
Name.isna().sum() #0 found

Name_values = Name.value_counts()
Name_dict = {'Mr':0, 
             'Miss':1, 
             'Mrs':1, 
             'Master':0, 
             'Rev':3, 
             'Dr':2, 
             'Col':3, 
             'Mlle':1, 
             'Ms':1, 
             'Major':3, 
             'Don':0, 
             'Mme':1, 
             'Capt':3, 
             'Sir':3, 
             'the Countess':1, 
             'Lady':1, 
             'Jonkheer':0, 
             'Dona':1, 
             }


Name = Name.apply(lambda x: Name_dict[x])

Name = Name.values.reshape((dataset.shape[0], 1))
Name = OneHotEncoder().fit_transform(Name).toarray()

#np.save('npy_files/Name.npy', Name)
#Name = np.load('npy_files/Name.npy')

del Name_values, Name_dict

##############################Final Dataset
dataset = np.concatenate((Age, Embarked, FamilySize, Fare, IsAlone, 
                          Name, Parch, Pclass, Sex, SibSp), axis = 1)

del Age, Embarked, FamilySize, Fare, IsAlone, Name, Parch, Pclass, Sex, SibSp

#np.save('npy_files/dataset.npy', dataset)
#dataset = np.load('npy_files/dataset.npy')



###############################Creating Train, CV and test Sets
X_train, X_test = dataset[:891, :], dataset[891:, :]

#CV created for testing performance -> Temporary
#X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size = 0.15, random_state = 32)

del dataset, missing

##############################Model -> XGB 

model = XGBClassifier(n_estimators = 325, 
                      n_jobs = 4, 
                      max_depth = 5, 
                      learning_rate = 0.05)

model.fit(X_train, Y_train)

Y_train_pred = model.predict(X_train)
#Y_cv_pred = model.predict(X_cv)
Y_test_pred = model.predict(X_test)

cm_train = 0
for i in range(891):
    if Y_train_pred[i] == Y_train[i][0]:
        cm_train+=1

#cm_cv = 0
#for i in range(134):
#    if Y_cv_pred[i] == Y_cv[i][0]:
#        cm_cv+=1

#Found that we can use 325-275 n_estimators for data

#print(cm_cv/134)
#print(cm_train/757)

#########################################














output = {'PassengerId': np.squeeze(test.iloc[:, 0:1].values),
          'Survived': Y_test_pred
          }
df = pd.DataFrame(output, columns = ['PassengerId', 'Survived'])
df.to_csv(r'Output.csv', index = None, header = True)