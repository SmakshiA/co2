import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

from flask import Flask, request,render_template

# Load the dataset
dataset = pd.read_csv("CO2emmission.csv")

# Split the dataset into input (X) and output (y) variables
#X = dataset.iloc[:, 7:10].values
X = dataset.iloc[:, [3,4,9]].values
y = dataset.iloc[:, -1].values

print(X)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a Linear Regression model and fit it to the training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = regressor.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", rmse)

print(mean_absolute_percentage_error(y_test, y_pred))
#  y_test, pred = np.array(y_test), np.array(pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test))
print(mape)


score = r2_score(y_test,y_pred)
#print("The accuracy of our model is {}%".format(round(score, 2) *100))

print(score*100)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_inputs', methods=['POST'])
def process_inputs():
    input1 = request.form['input1']
    input2 = request.form['input2']
    input3 = request.form['input3']

    input1 = float(input1)
    input2 = float(input2)
    input3 = float(input3)

    # Do something with the inputs
    print(f'Input 1: {input1}')
    print(f'Input 2: {input2}')
    print(f'Input 3: {input3}')

    y_pred = regressor.predict([[input1,input2,input3]])
    print(y_pred)

    return f"CO2 emmssion by your car is {y_pred} "

if __name__ == '__main__':
    app.run(debug=True)
