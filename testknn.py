import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import joblib

dataset = pd.read_csv('Student-Employability-Datasets.csv')

# scaling values into 0-1 range
scaler = MinMaxScaler(feature_range=(0, 1))
features = ['GENERAL_APPEARANCE','MANNER_OF_SPEAKING','PHYSICAL_CONDITION','MENTAL_ALERTNESS','SELF_CONFIDENCE','ABILITY_TO_PRESENT_IDEAS','COMMUNICATION_SKILLS','Student_Performance_Rating']
dataset[features] = scaler.fit_transform(dataset[features])

X = dataset[['GENERAL_APPEARANCE','MANNER_OF_SPEAKING','PHYSICAL_CONDITION','MENTAL_ALERTNESS','SELF_CONFIDENCE','ABILITY_TO_PRESENT_IDEAS','COMMUNICATION_SKILLS','Student_Performance_Rating']]
y = dataset['CLASS']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model = joblib.load('knnModel.pkl')
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print('=== K-Nearest Neighbor algorithm ===\n')
print('===Confusion matrix===')
print(cm)
print('TP:',cm[0][0])
print('TN:',cm[1][1])
print('FP:',cm[0][1])
print('FN:',cm[1][0])
print('===Classification report ( metrics )===')
print(classification_report(y_test,y_pred))
