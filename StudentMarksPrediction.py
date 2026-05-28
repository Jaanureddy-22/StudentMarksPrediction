import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score

# Load Dataset
data=pd.read_csv('Student_Marks.csv')

#check information

data.describe()
data.info()

# Features and Target
x=data[['time_study']]
y=data['Marks']

# visualisation

plt.scatter(x,y,marker='*')
plt.xlabel('study hours')
plt.ylabel('marks')
plt.title("study hours vs marks")
plt.show()

# Train Test Split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,random_state=40
)

# Model
model=LinearRegression()

# Train
model.fit(x_train,y_train)

# Predict Test Data
y_pred=model.predict(x_test)

# Accuracy
r2=r2_score(y_test,y_pred)

print(f'Model Accuracy: {r2*100:.2f}%')

# User Input
hours=float(input("Enter study hours: "))

# Prediction
input_data=pd.DataFrame([[hours]],columns=['time_study'])

prediction=model.predict(input_data)

print(f"Predicted Marks: {prediction[0]:.2f}")