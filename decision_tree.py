# The dataset was collected from different university agencies in the Philippines. 
# It consists of Mock job Interview Results of 2.982 observations. 
# The dataset collected needs to be normalized and cleaned. 
# The dataset that was collected is compliant with the Data Privacy Act of the Philippines.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

dataset = pd.read_csv('Student-Employability-Datasets.csv')

# scaling values into 0-1 range
scaler = MinMaxScaler(feature_range=(0, 1))
features = ['GENERAL_APPEARANCE','MANNER_OF_SPEAKING','PHYSICAL_CONDITION','MENTAL_ALERTNESS','SELF_CONFIDENCE','ABILITY_TO_PRESENT_IDEAS','COMMUNICATION_SKILLS','Student_Performance_Rating']
dataset[features] = scaler.fit_transform(dataset[features])

X = dataset[['GENERAL_APPEARANCE','MANNER_OF_SPEAKING','PHYSICAL_CONDITION','MENTAL_ALERTNESS','SELF_CONFIDENCE','ABILITY_TO_PRESENT_IDEAS','COMMUNICATION_SKILLS','Student_Performance_Rating']]
y = dataset['CLASS']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

joblib.dump(dt,'dtModel.pkl')

