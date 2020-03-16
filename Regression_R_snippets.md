# Modelos de regresion en R

## Dataset de prueba
```r
df_reg <- cars
colnames(df_reg) <- c('X', 'y')
head(df_reg)
dim(df_reg)
```

## Separacion Train / Test
```r
sample <- sample.int(n = nrow(df_reg), size = floor(.80*nrow(df_reg)), replace = F)
df_reg_train <- df_reg[sample, ]
df_reg_test  <- df_reg[-sample, ]
```

## Modelos

### Linear Model
Con `R base`
```r
reg_LM <- lm(y~X, data=df_reg_train)

#Resumen de parametros y ajuste
summary(reg_LM)
```
Con `caret`
```r
library(caret)

reg_LM <- train(y~X, data=df_reg_train, method='lm')

summary(reg_LM)
```

### Generalized Linear Model (GLM)
Parámetros adicionales:
* family = familia de funciones de distribucion a usar (y funcion link)  
Las opciones [aqui](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/family)

Con `R base`
```r
reg_GLM <- glm(formula=y~X, family=poisson(link = "log"), data=df_reg_train) #ejemplo con poisson
summary(reg_GLM)
```

Con `caret`
```r
library(caret)

reg_GLM2 <- train(y~X, data=df_reg_train, method='glm', family=poisson) #ejemplo con poisson
summary(reg_GLM2)
```

### k-nearest neighbors

Parámetros adicionales:
* k = número de vecinos cercanos a considerar  
Aqui ya se incluye una busqueda de parametros (tune grid)  
Adicionalmente conviente especificar el método de resampleo
* method = metodo de resampleo ('cv', 'repeatedcv', 'LOOCV', ...)
* number = número de particiones (k-fold)
* repeats = número de repeticiones (si es necesario)  

```r
library(caret)

reg_KNN <- train(y ~ X,
  data = df_reg_train,
  method = "knn",
  trControl = trainControl(method = "repeatedcv", number = 5, repeats=10), # Seleccionar metodo de resampling (#aqui: repeated 10-fold CV)
  tuneGrid = expand.grid(k = seq(1, 10))) # Generar la malla de valores de parámetros para buscar optimo

reg_KNN #ver resultados
plot(reg_KNN) #visualizacion gráfica
``` 

## Métricas
Generar predicción
```r
y_pred <- predict(reg_model, df_reg_test)
```

### Mean Absolute Error (MAE)
```r
mean(abs(df_reg_test$y - predict(reg_model, df_reg_test)))
```

### Mean Absolute Percentage Error (MAPE)

mean(abs(df_reg_test$y - predict(reg_LM, df_reg_test))/df_reg_test$y)






