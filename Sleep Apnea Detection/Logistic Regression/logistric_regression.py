import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

# Reading input data
data_input = pd.read_excel('datasheet.xlsx', usecols=['Neckcircum', 'BMI', 'STOPBANG_total ', 'ESS>11'])

# Reading output data
data_AHI = pd.read_excel('datasheet.xlsx', usecols=['AHI'])

# Cleaning the data
nan_mask = data_input['Neckcircum'].isna()
data_input = data_input[~nan_mask]
data_AHI_filtered = data_AHI[~nan_mask]


data_output = []

for value in data_AHI_filtered['AHI']:
    if(float(value) >= 15.0):
        data_output.append(1)
    
    else:
        data_output.append(0)


data_output = np.array(data_output)
data_input = np.array(data_input)


# Diving input into training set, validation and test set
X_train, X_temp, y_train, y_temp = train_test_split(data_input, data_output, test_size=0.30, random_state=44)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state= 44)


# Create the model
model = LogisticRegression()

model.fit(X_train, y_train)  # Train model

# Predictions and accuracy for validation data
y_pred_valid = model.predict(X_validation)
accuracy = metrics.accuracy_score(y_validation, y_pred_valid)
print("Validation Accuracy:", accuracy)

mse = metrics.mean_squared_error(y_validation, y_pred_valid)
print("Validation MSE:", mse)

# Model coefficients and intercept
theta1, theta2,theta3,theta4 = model.coef_[0]
y_intercept = model.intercept_[0]
print("Model Coefficients: theta1 =", theta1, ", theta2 =", theta2, ", theta3 = ", theta3, ", theta4 = ", theta4)
print("Y Intercept:", y_intercept)


# Predicting on test data
y_pred_test = model.predict(X_test)

# Accuracy of test data
test_accuracy = metrics.accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", test_accuracy)

# Mean Squared Error for test data
test_mse = metrics.mean_squared_error(y_test, y_pred_test)
print("Test MSE:", test_mse)


# Calculate the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred_test)

# Plot the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Sleep Apnea", "Sleep Apnea"], yticklabels=["No Sleep Apnea", "Sleep Apnea"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.show()