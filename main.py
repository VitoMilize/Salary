import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

salary_data = pd.read_csv('Salary_Data.csv')
X = salary_data.iloc[:, :-1]
y = salary_data.iloc[:, 1]

plt.figure(figsize=(36, 12))


plt.subplot(1, 3, 1)
sns.histplot(X['YearsExperience'], kde=False, bins=10)
plt.title('Histogram of YearsExperience')

plt.subplot(1, 3, 2)
sns.countplot(x=X['YearsExperience'])
plt.title('Countplot of YearsExperience')

plt.subplot(1, 3, 3)
sns.scatterplot(x=X['YearsExperience'], y=y)
plt.title('Scatterplot of YearsExperience vs. Salary')

plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(y_pred)

plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, lr.predict(X_train), color='red')
plt.title('Salary ~ Experience (Train set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, lr.predict(X_test), color='red')
plt.title('Salary ~ Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print('MAE :', metrics.mean_absolute_error(y_test, y_pred))
print('MSE :', metrics.mean_squared_error(y_test, y_pred))
print('RMSE :', np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))


