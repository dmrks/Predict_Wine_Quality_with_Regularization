import numpy as np
import pandas as pd
import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('wine_quality.csv')
print(df.columns)
y = df['quality']
features = df.drop(columns = ['quality'])


## 1. Data transformation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(features)
X = scaler.transform(features)

## 2. Train-test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 99)

## 3. Fit a logistic regression classifier without regularization
from sklearn.linear_model import LogisticRegression
clf_no_reg = LogisticRegression(penalty = 'none')
clf_no_reg.fit(x_train,y_train)

## 4. Plot the coefficients

predictors = features.columns
coefficients = clf_no_reg.coef_.ravel()
coef = pd.Series(coefficients,predictors).sort_values()
coef.plot(kind='bar', title = 'Coefficients (no regularization)')
plt.tight_layout()
plt.show()
plt.clf()

## 5. Training and test performance = 0.7266666666666667/ 0.7727598566308242

from sklearn.metrics import f1_score

y_pred_test = clf_no_reg.predict(x_test)
y_pred_train = clf_no_reg.predict(x_train)

print(f1_score(y_pred_test,y_test))
print(f1_score(y_pred_train,y_train))

## 6. Default Implementation (L2-regularized!)

clf_default = LogisticRegression()
clf_default.fit(x_train,y_train)

## 7. Ridge Scores same as 5

y_pred_test_def = clf_default.predict(x_test)
y_pred_train_def = clf_default.predict(x_train)

print(f1_score(y_pred_test_def,y_test))
print(f1_score(y_pred_train_def,y_train))

## 8. Coarse-grained hyperparameter tuning
training_array = []
test_array = []
C_array = [0.0001, 0.001, 0.01, 0.1, 1]

for i in C_array:
  clf_default = LogisticRegression(C = i)
  clf_default.fit(x_train,y_train)
  y_pred_test = clf_no_reg.predict(x_test)
  y_pred_train = clf_no_reg.predict(x_train)
  training_array.append(f1_score(y_pred_train_def,y_train))
  test_array.append(f1_score(y_pred_test_def,y_test))


## 9. Plot training and test scores as a function of C
plt.plot(C_array,training_array)
plt.plot(C_array,test_array)
plt.xscale('log')
plt.show()
plt.clf()

## 10. Making a parameter grid for GridSearchCV
C_array=np.logspace(-4, -2, 100)
tuning_C = {'C':C_array}

## 11. Implementing GridSearchCV with l2 penalty
from sklearn.model_selection import GridSearchCV

clf = LogisticRegression()
gs = GridSearchCV(clf, param_grid = tuning_C, scoring ='f1', cv =5)

gs.fit(x_train,y_train)

## 12. Optimal C value and the score corresponding to it = {'C': 1.0002302850208247} // 0.7687282512938214
print(gs.best_params_)
print(gs.best_score_)

## 13. Validating the "best classifier"
clf_best_ridge = LogisticRegression(C = gs.best_params_['C'])
clf_best_ridge.fit(x_train,y_train)
pred_test = clf_best_ridge.predict(x_test)
y_pred_best = clf_best_ridge.predict(x_test)
print(f1_score(y_test,y_pred_best))

## 14. Implement L1 hyperparameter tuning with LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV

clf_l1 = LogisticRegressionCV(Cs=np.logspace(-2,2, 100),penalty='l1', solver ='liblinear',scoring='f1', cv=5)
clf_l1.fit(X,y)

## 15. Optimal C value and corresponding coefficients = Best C [1.51991108] 
#Best Coefficients [[ 0.17493577 -0.51528419 -0.14693451  0.12432199 -0.2099992   0.2393345 -0.54520332 -0.10848853 -0.06899449  0.44992835  0.90100081]]

print("Best C",clf_l1.C_)
print("Best Coefficients",clf_l1.coef_)

## 16. Plotting the tuned L1 coefficients

coefficients = clf_l1.coef_.ravel()
coef = pd.Series(coefficients,predictors).sort_values()
 
plt.figure(figsize = (12,8))
coef.plot(kind='bar', title = 'Coefficients for tuned L1')
plt.tight_layout()
plt.show()
plt.clf()
