# Starting Point
```python
df = pd.read_csv("house_prices.csv")
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df=df[df["TotalSF"]<6000]
X = df[['TotalSF']] # pandas DataFrame
y = df["SalePrice"] # pandas Series
```
# Modelos
## Linear regression
```python
from sklearn.linear_model import LinearRegression

#Crear instancia del modelo
reg_LR = LinearRegression()

# Ajustar modelo
reg_LR.fit(X_train,y_train)

# Valores
reg_LR.coef_ #Pendiente
reg_LR.intercept_ #Punto de corte
```

## k-neighbors
```python
from sklearn.neighbors import KNeighborsRegressor

# Crear instancia
reg_knn = KNeighborsRegressor(n_neighbors=2)

# Ajustar modelo
reg_knn.fit(X_train,y_train)
```
## Decision tree

```python
from sklearn.tree import DecisionTreeRegressor

# Crear instancia
reg_DT = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20) #cambiar parametros

# Ajustar modelo
reg_DT.fit(X_train, y_train)
```
# Metrics
```python
# Prediccion
y_pred = reg_model.predict(X_test) #se puede incluir directamente así en los calculos
```
## Mean Absolute Error (MAE)
```python
from sklearn.metrics import mean_absolute_error

MAE_reg_model = mean_absolute_error(y_pred, y_test)
```
## Mean Absolute Percentage Error (MAPE)
```python
import numpy as np

# Estimacion con numpy
np.mean(np.abs(y_pred - y_test)/y_test)
```

## Mean Squared Error (MSE)
```python
from sklearn.metrics import mean_squared_error

mean_squared_error(y_pred, y_test)
```

## Root Mean Squared Error (RMSE)
```python
from sklearn.metrics import mean_squared_error
import numpy as np

np.sqrt(mean_squared_error(y_pred, y_test))
```

## Correlation
```python
# Direct Calculation
np.corrcoef(y_pred,y_test)[0][1]

# Custom Scorer
from sklearn.metrics import make_scorer
def corr(y_pred,y_test):
return np.corrcoef(y_pred,y_test)[0][1]

# Put the scorer in cross_val_score
cross_val_score(reg_model,X,y,cv=5,scoring=make_scorer(corr))
```

## Bias
```python
# Direct Calculation
np.mean(y_pred - y_test)

# Custom Scorer
from sklearn.metrics import make_scorer
def bias(y_pred,y_test):
return np.mean(y_pred-y_test)

# Put the scorer in cross_val_score
cross_val_score(reg,X,y,cv=5,scoring=make_scorer(bias))
```

# Evaluation
## Train - Test split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
```
## Cross Validation
## Grid Search
## Randomized Search
