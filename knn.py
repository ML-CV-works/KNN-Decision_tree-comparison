import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

dataset = pd.read_csv('Student-Employability-Datasets.csv')

# scaling values into 0-1 range
scaler = MinMaxScaler(feature_range=(0, 1))
features = ['GENERAL_APPEARANCE','MANNER_OF_SPEAKING','PHYSICAL_CONDITION','MENTAL_ALERTNESS','SELF_CONFIDENCE','ABILITY_TO_PRESENT_IDEAS','COMMUNICATION_SKILLS','Student_Performance_Rating']
dataset[features] = scaler.fit_transform(dataset[features])

X = dataset[['GENERAL_APPEARANCE','MANNER_OF_SPEAKING','PHYSICAL_CONDITION','MENTAL_ALERTNESS','SELF_CONFIDENCE','ABILITY_TO_PRESENT_IDEAS','COMMUNICATION_SKILLS','Student_Performance_Rating']]
y = dataset['CLASS']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)
joblib.dump(knn,'knnModel.pkl')