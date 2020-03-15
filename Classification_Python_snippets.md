# Starting point
Ejemplo
```python
df = pd.read_csv("terrain.csv")

X = df[["bumpiness","grade"]].values # los predictores, como data frame
y = df["target"] # variable objetivo, como serie
```

# Modelos

## Logistic regression

```python
from sklearn.linear_model import LogisticRegression

# Crear instancia
clf_LogR = LogisticRegression()

# Ajustar modelo
clf_LogR.fit(X_train,y_train)
```

## k-nearest neighbors
Parámetros principales
* n_neighbors: vecinos más próximos a considerar

```python
from sklearn.neighbors import KNeighborsClassifier

# Crear instancia
clf_knn = KNeighborsClassifier(n_neighbors=2)

# Ajustar modelo
clf_knn.fit(X_train,y_train)
```

Parámetros principales
* Max_depth: Número máximo de cortes
* Min_samples_leaf: Número mínimo de observaciones en cada subgrupo (hoja)

```python
from sklearn.tree import DecisionTreeRegressor

# Crear instancia
reg_DT = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20) #cambiar parametros

# Ajustar modelo
reg_DT.fit(X_train, y_train)
```

## Decision Tree

Parámetros principales
* Max_depth: Número máximo de cortes
* Min_samples_leaf: Número mínimo de observaciones en cada subgrupo (hoja)

```python
from sklearn.tree import DecisionTreeClassifier

# Crear instancia
clf_DT = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20) #cambiar parametros

# Ajustar modelo
clf_DT.fit(X_train, y_train)
```

## Support Vector Machine

Parámetros principales:
* C: Suma del error de los margenes
* kernel: tipo de plano de separacion (linear / rbf / poly)
* linear: separacion linear 

* rbf: separacion circular
* Parámetro adicionl: gamma = Inversa del radio del circulo  

* poly: separacion por linea curva (polinomial)
* Parámetro adicional: degree = Grado del polinomio

```python
from sklearn.svm import SVC

# Crear instancia
clf_SVM = SVC(kernel="linear",C=10)

# Ajustar modelo
clf_SVM.fit(X,y)
```

# Random Forest
Prámetros: ver decision tree

```python
from sklearn.ensemble import RandomForestClassifier

# Crear instancia
clf_RF = RandomForestClassifier(max_depth=4)

# Ajustar modelo
clf_RF.fit(X,y)
```

# Gradient Boosting Tree
Parámetros: ver decision tree

```python
from sklearn.ensemble import GradientBoostingClassifier

# Crear instancia
clf_GBT = GradientBoostingClassifier(max_depth=4)
# Fit the data
clf_GBT.fit(X,y)
```
#Métricas

Generar prediccion para evaluar
```python
y_pred = reg_model.predict(X_test) #se puede incluir directamente así en los calculos
```

## Accuracy
```python

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
```

## Precision and Recall
```python

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

precision_score(y_test,y_pred)

classification_report(y_test,y_pred)

```
## ROC curve
```python

from sklearn.metrics import roc_curve

# Elegir valor (clase) objetivo (0 o 1 en un caso binario) 
target_pos = 1 

fp,tp,_ = roc_curve(y_test,pred[:,target_pos])

plt.plot(fp,tp)
```
## AUC

```python

from sklearn.metrics import roc_curve, auc

fp,tp,_ = roc_curve(y_test,pred[:,1])
auc(fp,tp)

```

# Evaluación
## Train - Test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
```


## Cross Validation
Parámetros principales
* cv = número de particiones entre las que combinar
* scoring = métrica de evaluacion  
Métricas de Scoring [aqui](https://scikit-learn.org/stable/modules/model_evaluation.html)  
Además se pueden añadir los custom (ver Bias / Correlation)
```python
from sklearn.model_selection import cross_val_score

cross_val_score(clf_model,X,y, cv=5, scoring='accuracy')

# Para obtener el valor promedio
import numpy as np
np.mean(cross_val_score(reg_model,X,y, cv=5, scoring='accuracy'))
```

## Grid Search
Parámetros principales:
* Modelo a usar (sin especificar parametros)
* param_grid = `diccionario` con los nombres de los parametros del modelo (en keys) y los posibles valores (en values)
* cv = particiones de cross validation
* scoring = métrica para evaluar qué modelo es mejor (ver cross validation)

```python
from sklearn.model_selection import GridSearchCV

clf_knn_test = GridSearchCV(KNeighborsRegressor(), #Ejemplo con KNN
                         param_grid = {'n_neighbors':np.arange(2,50)},
                        cv=5,
                        scoring='neg_mean_absolute_error')
```

## Randomized Search
Parámetros principales:
* Modelo a usar (sin especificar parametros)
* param_grid = `diccionario` con los nombres de los parametros del modelo (en keys) y los posibles valores (en values)
* cv = particiones de cross validation
* scoring = métrica para evaluar qué modelo es mejor (ver cross validation)
* n_iter = número de pruebas aleatorias a realizar entre todas las posibles
```python
from sklearn.model_selection import RandomizedSearchCV

clf_DT_testR = RandomizedSearchCV(DecisionTreeClassifier(), #Ejemplo con Decision Tree
                                param_distributions={'max_depth':[2,3,5,10],
                                                    'min_samples_leaf':[5,10,15,20,30,40]},
                                cv=5,
                                scoring='neg_mean_absolute_error',
                                n_iter=10)
```
