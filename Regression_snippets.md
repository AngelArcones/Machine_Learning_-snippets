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
reg = LinearRegression()

# Ajustar modelo
reg.fit(X,y)

# Valores
reg.coef_ #Pendiente
reg.intercept_ #Punto de corte
```

## k-neighbors
```python
from sklearn.neighbors import KNeighborsRegressor

# Crear instancia
regk = KNeighborsRegressor(n_neighbors=2)

# Ajustar modelo
regk.fit(X_train,y_train)
```
## Decision tree

# Metrics
## Mean Absolute Error (MAE)
```python
from sklearn.metrics import mean_absolute_error

MAE_regLin = mean_absolute_error(reg.predict(X_test), y_test)
```
## Mean Absolute Percentage Error (MAPE)
## Root Mean Squared Error (RMSE)
## Correlation
## Bias

#Evaluation
## Train - Test split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
```
## Cross Validation
## Grid Search
## Randomized Search
