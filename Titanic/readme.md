Titanic: Machine Learning from Disaster

FINAL Score: 0.8451 (or 84.51%)

url: https://www.kaggle.com/c/titanic/overview

Challenge Info: The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

Methodology used:

Implemented using python programming language, modules used are pandas, numpy, scikit-learn and xgboost.

#########################################################################################
Column: Pclass, 
        SibSp, 
        Parch, 
        -> direct Data

Column: Sex -> applied LabelEncoder

Column: Fare -> applied median on missing values and StandardScaler

Column: Embarked -> applied mode on missing values and StandardScaler and OneHotEncoder

Column: Age -> applied mean on missing values and StandardScaler

Column: FamilySize -> Feature Engineering from SibSp+Parch+1

Column: IsAlone -> Feature Engineering from FamilySize -> 1 if FamilySize>1 else 0

Column: Name -> applied various data cleaning methods and manually OneHotEncoded data

#########################################################################################

Applied XGBRegressor with max_depth = 5 and n_estimators = 375

FINAL Score: 0.8451 (or 84.51%)

