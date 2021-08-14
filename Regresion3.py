# Autores:
#     - Joaquín René BECK
#     - Leandro FLASCHKA
#     - Carlos Ernesto CORTÉS PARRA
#     {joaquinrbeck, leoflaschka, cecortes}@gmail.com

# Agosto 2021

#Manipulación de datos
import pandas as pd
pd.options.display.max_columns = 20
import numpy as np

#Visualización
import matplotlib.pyplot as plt
import seaborn as sns

#Estadísticas
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import  mean_absolute_error,mean_squared_error,r2_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import  GridSearchCV

# Read in data
print("Leyendo dataset...")
df = pd.read_csv('Dataset//houses_to_rent_v2.csv', encoding='utf-8')

print('\n <<< Cantidad de ejemplos y características >>>')
print(df.shape)

print('\n <<< Primeras filas del dataset >>>')
print(df.head(20))

print('\n <<< Información del dataset >>>')
print(df.info())

print('\n <<< Información estadística del dataset >>>')
print(df.describe(include='all'))

print('\n <<< Ciudades >>>')
cities = df['city'].unique()
# print(cities)

###############################################################################
# DAE breve (Data Analysis Exploration)
###############################################################################
print('\n <<< Histograma de renta en', *cities, ' >>>')
plt.figure(figsize=(12, 6))
titulo = 'Histograma de renta en: ', *cities
plt.title(titulo)
sns.kdeplot(df['rent amount (R$)'])
plt.xticks(np.arange(df['rent amount (R$)'].min(), df['rent amount (R$)'].max(), step=2500));
plt.grid(True)
plt.show()

###############################################################################
# Correlaciones para determinar las variables de mayor peso
###############################################################################
print('\n <<< Correlaciones >>>')
plt.figure(figsize=(9, 8))
numData = df._get_numeric_data()    #Considera solo las columnas con datos numéricos
var_num_corr = numData.corr()       #Calcula la correlación por pares de series
print(var_num_corr)
cmap = sns.diverging_palette(230,20,as_cmap=True)
mask = np.triu(var_num_corr)
sns.heatmap(var_num_corr,linewidth=0.01, linecolor='white', center=0, vmin=-1, vmax=1, annot=True, cmap=cmap)
plt.suptitle('Mapa de Correlaciones',fontsize=16)
plt.show()

###############################################################################
#"""Se realiza un diagrama de cajas por ciudades"""
###############################################################################
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['city'], y=df['rent amount (R$)'])
plt.suptitle('Distribución de la renta por ciudades',fontsize=16)
plt.show()

###############################################################################
#Selección de columnnas con mayor correlación
###############################################################################
cols = ['city', 'rooms', 'bathroom', 'parking spaces', 'fire insurance (R$)', 'furniture']
x = df[cols]
y = df['rent amount (R$)']

labelencoder = LabelEncoder()   #Crea el objeto codificador de etiquetas
x.loc[:, 'furniture'] = labelencoder.fit_transform(x.loc[:, 'furniture'])
#print(x)

# print('\n <<< Conversión categórica de ciudades >>>')
dummy = pd.get_dummies(x, columns=['city'])
dummy.drop(columns=['city_Belo Horizonte'], inplace=True)
x = dummy
#print(x)

###############################################################################
# Dividir en train y test
###############################################################################
test_size = 0.3
X_train, x_test, y_train, y_test = train_test_split(x, y,test_size=test_size, random_state = 123)

###############################################################################
# Validación cruzada para RandomForestRegressor con cross_val_score
###############################################################################
print('\n <<< k-cross-validation Random Forest Regressor >>>')
train_scores=[]
cv_scores=[]
train_scores_test=[]
# Valores evaluados
stop=200
estimator_range=range(1,stop,5)
start_time = pd.Timestamp.now()
for n_estimators in estimator_range:
   print("%0.0f" % (n_estimators*100/stop) + "%\r",end="")
   model = RandomForestRegressor(n_estimators=n_estimators,random_state=123)
   model.fit(X_train, y_train)
   prediction=model.predict(X_train)
   rmse=mean_squared_error(y_true=y_train, y_pred=prediction,squared=False)
   train_scores.append(rmse)

   prediction_test = model.predict(x_test)
   rmse_test=mean_squared_error(y_true=y_test, y_pred=prediction_test,squared=False)
   train_scores_test.append(rmse_test)

   # Error de validación cruzada
   scores=cross_val_score(estimator=model,X=X_train,y=y_train,scoring='neg_root_mean_squared_error',cv=10)
   cv_scores.append(-1*scores.mean())

runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)
#Gráfico
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(estimator_range, train_scores, label="train scores")

ax.plot(estimator_range, train_scores_test, label="test scores")

ax.plot(estimator_range, cv_scores, label="cv scores")
ax.plot(estimator_range[np.argmin(cv_scores)], min(cv_scores),
       marker='o', color = "red", label="min score")
ax.set_ylabel("RMSE")
ax.set_xlabel(f"n_estimators. Óptimo: {estimator_range[np.argmin(cv_scores)]}")
ax.set_title("Cross validation Error vs Número árboles - Random Forest Regressor")
plt.legend();
print(f"n_estimators óptimo: {estimator_range[np.argmin(cv_scores)]}")
plt.show()

###############################################################################
# Validación cruzada para SVR con cross_val_score kernel 'rbf'
# variando C. Epsilon por defecto 0.1
###############################################################################
kernel='rbf'
print('\n <<< k-cross-validation SVR kernel "'+ kernel + '" >>>')
train_scores=[]
cv_scores=[]
train_scores_test=[]
# Valores evaluados
stop=100
estimator_range=range(1,stop,5)
kernel='rbf'
start_time = pd.Timestamp.now()
for n_estimators in estimator_range:
    print("%0.0f" % (n_estimators*100/stop) + "%\r",end="")
    model = SVR(kernel=kernel,C=n_estimators,)
    model.fit(X_train, y_train)
    prediction=model.predict(X_train)
    rmse=mean_squared_error(y_true=y_train, y_pred=prediction,squared=False)
    train_scores.append(rmse)

    prediction_test = model.predict(x_test)
    rmse_test=mean_squared_error(y_true=y_test, y_pred=prediction_test,squared=False)
    train_scores_test.append(rmse_test)

    # Error de validación cruzada
    scores=cross_val_score(estimator=model,X=X_train,y=y_train,scoring='neg_root_mean_squared_error',cv=5)
    cv_scores.append(-1*scores.mean())
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)
#Gráfico
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(estimator_range, train_scores, label="train scores")
ax.plot(estimator_range, train_scores_test, label="test scores")
ax.plot(estimator_range, cv_scores, label="cv scores")
ax.plot(estimator_range[np.argmin(cv_scores)], min(cv_scores), marker='o', color = "red", label="min score")
ax.set_ylabel("RMSE")
ax.set_xlabel(f"C. Óptimo: {estimator_range[np.argmin(cv_scores)]}")
ax.set_title("Cross validation Error vs C - SVR kernel " + kernel)
plt.legend();
print(f"C óptimo: {estimator_range[np.argmin(cv_scores)]}")
plt.show()

###############################################################################
# Validación cruzada para SVR con cross_val_score kernel 'poly'
# variando C. Epsilon por defecto 0.1
###############################################################################
kernel='poly'
print('\n <<< k-cross-validation SVR kernel "'+ kernel + '" >>>')
train_scores=[]
cv_scores=[]
train_scores_test=[]
# Valores evaluados
stop=100
estimator_range=range(1,stop,5)
start_time = pd.Timestamp.now()
for n_estimators in estimator_range:
    print("%0.0f" % (n_estimators*100/stop) + "%\r",end="")
    model = SVR(kernel=kernel,C=n_estimators,)
    model.fit(X_train, y_train)
    prediction=model.predict(X_train)
    rmse=mean_squared_error(y_true=y_train, y_pred=prediction,squared=False)
    train_scores.append(rmse)

    prediction_test = model.predict(x_test)
    rmse_test=mean_squared_error(y_true=y_test, y_pred=prediction_test,squared=False)
    train_scores_test.append(rmse_test)

    # Error de validación cruzada
    scores=cross_val_score(estimator=model,X=X_train,y=y_train,scoring='neg_root_mean_squared_error',cv=5)
    cv_scores.append(-1*scores.mean())
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)
#Gráfico
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(estimator_range, train_scores, label="train scores")
ax.plot(estimator_range, train_scores_test, label="test scores")
ax.plot(estimator_range, cv_scores, label="cv scores")
ax.plot(estimator_range[np.argmin(cv_scores)], min(cv_scores), marker='o', color = "red", label="min score")
ax.set_ylabel("RMSE")
ax.set_xlabel(f"C. Óptimo: {estimator_range[np.argmin(cv_scores)]}")
ax.set_title("Cross validation Error vs C - SVR kernel " + kernel)
plt.legend();
print(f"C óptimo: {estimator_range[np.argmin(cv_scores)]}")
plt.show()

###############################################################################
# Validación cruzada para SVR con cross_val_score kernel 'linear'
# variando C. Epsilon por defecto 0.1
###############################################################################
kernel='linear'
print('\n <<< k-cross-validation SVR kernel "'+ kernel + '" >>>')
train_scores=[]
cv_scores=[]
train_scores_test=[]
# Valores evaluados
stop=100
estimator_range=range(1,stop,5)
start_time = pd.Timestamp.now()
for n_estimators in estimator_range:
    print("%0.0f" % (n_estimators*100/stop) + "%\r",end="")
    model = SVR(kernel=kernel,C=n_estimators,)
    model.fit(X_train, y_train)
    prediction=model.predict(X_train)
    rmse=mean_squared_error(y_true=y_train, y_pred=prediction,squared=False)
    train_scores.append(rmse)

    prediction_test = model.predict(x_test)
    rmse_test=mean_squared_error(y_true=y_test, y_pred=prediction_test,squared=False)
    train_scores_test.append(rmse_test)

    # Error de validación cruzada
    scores=cross_val_score(estimator=model,X=X_train,y=y_train,scoring='neg_root_mean_squared_error',cv=5)
    cv_scores.append(-1*scores.mean())
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)
#Gráfico
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(estimator_range, train_scores, label="train scores")
ax.plot(estimator_range, train_scores_test, label="test scores")
ax.plot(estimator_range, cv_scores, label="cv scores")
ax.plot(estimator_range[np.argmin(cv_scores)], min(cv_scores), marker='o', color = "red", label="min score")
ax.set_ylabel("RMSE")
ax.set_xlabel(f"C. Óptimo: {estimator_range[np.argmin(cv_scores)]}")
ax.set_title("Cross validation Error vs C - SVR kernel " + kernel)
plt.legend();
print(f"C óptimo: {estimator_range[np.argmin(cv_scores)]}")
plt.show()

###############################################################################
# Validación cruzada para SVR con GridSearchCV
###############################################################################
# Grid con los valores de costo a evaluar
# grid_hiperparametros = {'C': [0.001, 0.01, 0.1, 0.3, 0.5, 1, 2, 3, 4, 5, 7, 10],
#                         'gamma': [0.01, 0.1, 1, 5, 10]}
print('\n <<< GridSearchCV - SVR >>>')
grid_hiperparametros = {'C': [1.1, 5.4, 170, 1001],
                        'epsilon':[0.0003, 0.007, 0.0109, 0.019, 0.14, 0.05, 8, 0.2, 3, 2, 7],
                        'gamma': [0.7001, 0.008, 0.001, 3.1, 1, 1.3, 5]}
# Definimos la busqueda mediante 10-fold CV
busqueda_grid = GridSearchCV(estimator = SVR(kernel='rbf'),
                             param_grid = grid_hiperparametros,
                             cv = 5, #folds
                             scoring='neg_mean_squared_error',
                             #refit = 'accuracy',
                             return_train_score = True,
                             n_jobs = -1,
                             verbose=2)


# busqueda
busqueda_grid.fit(X_train, y_train)

#Valores de la búsqueda
print(busqueda_grid.cv_results_.keys())
# dict_keys(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
# 'param_C', 'param_gamma', 'params', 'split0_test_score', 'split1_test_score',
# 'split2_test_score', 'split3_test_score', 'split4_test_score', 'split5_test_score',
# 'split6_test_score', 'split7_test_score', 'split8_test_score', 'split9_test_score',
# 'mean_test_score', 'std_test_score', 'rank_test_score', 'split0_train_score',
# 'split1_train_score', 'split2_train_score', 'split3_train_score', 'split4_train_score',
# 'split5_train_score', 'split6_train_score', 'split7_train_score', 'split8_train_score',
# 'split9_train_score', 'mean_train_score', 'std_train_score'])

# Almacenamos en variables los resultados de la busqueda grid
C_cv = busqueda_grid.cv_results_['param_C'].tolist()
gamma_cv = busqueda_grid.cv_results_['param_gamma'].tolist()

accuracy_medio_cv = busqueda_grid.cv_results_['mean_test_score'].tolist()
accuracy_std_cv = busqueda_grid.cv_results_['std_test_score'].tolist()

# data frame con los resultados de la validacion cruzada
df_resultados_cv = pd.DataFrame({'C': C_cv,
                                 'gamma': gamma_cv,
                                 'mean_accuracy': accuracy_medio_cv,
                                 'std_accuracy': accuracy_std_cv})

print(df_resultados_cv)
#
# # Pivotamos el df para obtener los datos en tres columnas: C, accuracy con degree2 y accuracy con degree3
# # df_resultados_cv.pivot(index = 'C', columns = 'gamma', values = 'mean_accuracy').plot()
# # plt.figure(figsize=(15,5))
# # plt.xlabel('Costo (C)')
# # plt.ylabel('Accuracy')
# # plt.title('Modelo SVR (rbf): accuracy ~ C y gamma (10-fold cv)')
# # plt.show()

# Mejores hiperparametros
print('Valor de costo y gamma óptimos:', busqueda_grid.best_params_,
     '\nAccuracy asociado +- std:', round(busqueda_grid.cv_results_['mean_test_score'][busqueda_grid.best_index_],3),
     '+-', round(busqueda_grid.cv_results_['std_test_score'][busqueda_grid.best_index_],3))

###############################################################################
# Entrenamiento y predicción de SV Regressor GridSearchCV
###############################################################################
print('\n <<< Predicción SVR kernel=rbf,C=1001,epsilon=7,gamma=0.001 >>>')
metrica=[]
model=SVR(kernel='rbf',C=1001,epsilon=7,gamma=0.001)
start_time = pd.Timestamp.now()
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['Support Vector Regressor',mae_model,mse_model,r2_model,runing_time])

plt.figure(figsize=(15,5))
plt.title('Predicción SVR - kernel=rbf,C=1001,epsilon=7,gamma=0.001')
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="c",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.legend()
plt.grid(True)

# Dataframe con Métricas
metrica=pd.DataFrame(data=metrica,columns=['modelo','MAE','MSE','R2','Tiempo (s)'])
print(metrica)

plt.show()

###############################################################################
###############################################################################
# Validación cruzada para Decision Tree Regressor con GridSearchCV
###############################################################################
print('\n <<< GridSearchCV - DecisionTreeRegressor >>>')
grid_hiperparametros = {"criterion": ["mse", "mae"],
                        "min_samples_split": [10, 20, 40],
                        "max_depth": [2, 6, 8],
                        "min_samples_leaf": [20, 40, 100],
                        "max_leaf_nodes": [5, 20, 100],}
# Definimos la busqueda mediante 5-fold CV
busqueda_grid = GridSearchCV(estimator = DecisionTreeRegressor(random_state=0),
                             param_grid = grid_hiperparametros,
                             cv = 5, #folds
                             scoring='neg_mean_squared_error',
                             #refit = 'accuracy',
                             return_train_score = True,
                             n_jobs = -1,
                             verbose=2)

# busqueda
busqueda_grid.fit(X_train, y_train)

print('Parámetros DTR')
print(busqueda_grid.best_params_)
# {'criterion': 'mse', 'max_depth': 8, 'max_leaf_nodes': 100, 'min_samples_leaf': 20, 'min_samples_split': 10}

###############################################################################
# Entrenamiento y predicción de DecisionTreeRegressor GridSearchCV
###############################################################################
print('\n <<< Predicción Decision Tree Regressor (GridSearchCV) >>>')
metrica=[]
model=DecisionTreeRegressor(criterion='mse',max_depth=8,max_leaf_nodes=100,min_samples_leaf=20,min_samples_split=10)
start_time = pd.Timestamp.now()
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['Decision Tree Regressor',mae_model,mse_model,r2_model,runing_time])

plt.figure(figsize=(15,5))
plt.title('Predicción Decision Tree Regressor (GridSearchCV)')
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="g",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.legend()
plt.grid(True)

# Dataframe con Métricas
metrica=pd.DataFrame(data=metrica,columns=['modelo','MAE','MSE','R2','Tiempo (s)'])
print(metrica)

plt.show()


###############################################################################
###############################################################################
# Algoritmos con parámetros por default
###############################################################################
metrica=[]
###############################################################################
# Entrenamiento y predicción de Random Forest Regressor
###############################################################################
print('\n <<< Random Forest Regressor >>>')
start_time = pd.Timestamp.now()
model=RandomForestRegressor(random_state=123)
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['Random Forest Regressor',mae_model,mse_model,r2_model,runing_time])

plt.figure(figsize=(15,5))
plt.suptitle('Predicciones con parámetros por default',size=14)

plt.subplot(1,3,1)
plt.title('Random Forest Regressor')
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="b",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.legend()
plt.grid(True)

print('\n <<< Predicción de un valor [3,2,1,22,1,0,0,1,0] cuyo precio es 1650>>>')
prediction1 = model.predict([[3,2,1,22,1,0,0,1,0]])
print(prediction1)

###############################################################################
# Entrenamiento y predicción de Decision Tree Regressor
###############################################################################
print('\n <<< Predicción Decision Tree Regressor >>>')
start_time = pd.Timestamp.now()
model=DecisionTreeRegressor()
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['Decision Tree Regressor',mae_model,mse_model,r2_model,runing_time])

plt.subplot(1,3,2)
plt.title('Predicción Decision Tree')
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="g",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.legend()
plt.grid(True)

print('\n <<< Predicción de un valor [3,2,1,22,1,0,0,1,0] cuyo precio es 1650>>>')
prediction1 = model.predict([[3,2,1,22,1,0,0,1,0]])
print(prediction1)

###############################################################################
# Entrenamiento y predicción de SV Regressor
###############################################################################
print('\n <<< Predicción SVR >>>')
model=SVR()
start_time = pd.Timestamp.now()
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['Support Vector Regressor',mae_model,mse_model,r2_model,runing_time])

plt.subplot(1,3,3)
plt.title('Predicción SVR')
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="c",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.legend()
plt.grid(True)

print('\n <<< Predicción de un valor [3,2,1,22,1,0,0,1,0] cuyo precio es 1650>>>')
prediction1 = model.predict([[3,2,1,22,1,0,0,1,0]])
print(prediction1)

# Dataframe con Métricas
metrica=pd.DataFrame(data=metrica,columns=['modelo','MAE','MSE','R2','Tiempo (s)'])
print(metrica)

plt.tight_layout(pad=1)
plt.show()

###############################################################################
###############################################################################
# Entrenamiento y predicción de Random Forest Regressor n_estimators=166
###############################################################################
metrica=[]
print('\n <<< Random Forest Regressor n_estimators=166 >>>')
start_time = pd.Timestamp.now()
model=RandomForestRegressor(n_estimators=166, random_state=123)
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['Random Forest Regressor',mae_model,mse_model,r2_model,runing_time])

# plt.figure(figsize=(15,5))

plt.figure(figsize=(12,6))
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="b",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.title('Random Forest Regressor n_estimators=166')
plt.legend()
plt.grid(True)
plt.show()
# Dataframe con Métricas
metrica=pd.DataFrame(data=metrica,columns=['modelo','MAE','MSE','R2','Tiempo (s)'])
print(metrica)

###############################################################################
###############################################################################
# Entrenamiento y predicción de Support Vectort Regressor ajutado
###############################################################################
metrica=[]
kernel='rbf'
C=96
print('\n <<< Predicción SVR kernel='+ kernel +', C='+ str(C) + ' >>>')
model=SVR(kernel = kernel,C=C)
start_time = pd.Timestamp.now()
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['kernel='+ kernel +', C='+ str(C),mae_model,mse_model,r2_model,runing_time])

plt.figure(figsize=(15,5))
plt.suptitle('SVR Ajustado',size=14)
plt.tight_layout(pad=0.5)

plt.subplot(1,3,1)
plt.title('SVR '+ kernel +', C='+ str(C))
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="c",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.legend()
plt.grid(True)

kernel='poly'
C=96
print('\n <<< Predicción SVR kernel='+ kernel +', C='+ str(C) + ' >>>')
model=SVR(kernel = kernel,C=C)
start_time = pd.Timestamp.now()
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['kernel='+ kernel +', C='+ str(C),mae_model,mse_model,r2_model,runing_time])

plt.subplot(1,3,2)
plt.title('SVR '+ kernel +', C='+ str(C))
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="c",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.legend()
plt.grid(True)

kernel='linear'
C=46
print('\n <<< Predicción SVR kernel='+ kernel +', C='+ str(C) + ' >>>')
model=SVR(kernel = kernel,C=C)
start_time = pd.Timestamp.now()
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['kernel='+ kernel +', C='+ str(C),mae_model,mse_model,r2_model,runing_time])

plt.subplot(1,3,3)
plt.title('SVR '+ kernel +', C='+ str(C))
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="c",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.legend()
plt.grid(True)

# Dataframe con Métricas
metrica=pd.DataFrame(data=metrica,columns=['param','MAE','MSE','R2','Tiempo (s)'])
print(metrica)
plt.show()

###############################################################################
###############################################################################
# Entrenamiento y predicción ajustados
###############################################################################
metrica=[]
###############################################################################
# Entrenamiento y predicción de Random Forest Regressor
###############################################################################
print('\n <<< Random Forest Regressor >>>')
start_time = pd.Timestamp.now()
model=RandomForestRegressor(n_estimators=166, random_state=123)
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['Random Forest Regressor',mae_model,mse_model,r2_model,runing_time])

plt.figure(figsize=(15,5))
plt.suptitle('Predicciones con ajustes',size=14)

plt.subplot(1,3,1)
plt.title('Random Forest Regressor - n_estimators=166')
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="b",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.legend()
plt.grid(True)

###############################################################################
# Entrenamiento y predicción de Decision Tree Regressor
###############################################################################
# print('\n <<< Predicción Decision Tree Regressor >>>')
# start_time = pd.Timestamp.now()
# model=DecisionTreeRegressor()
# model.fit(X_train, y_train)
# prediction = model.predict(x_test)
# mae_model=mean_absolute_error(y_test,prediction)
# mse_model=np.sqrt(mean_squared_error(y_test, prediction))
# r2_model=r2_score(y_test,prediction)
# print('Tamaño Test: {:.0%}'.format(test_size))
# print('MAE:',mae_model)
# print('MSE:',mse_model)
# print('R2:',r2_model)
# runing_time = (pd.Timestamp.now() - start_time).total_seconds()
# print('tiempo de ejecución:',runing_time)
#
# metrica.append(['Decision Tree Regressor',mae_model,mse_model,r2_model,runing_time])
#
# plt.subplot(1,3,2)
# plt.title('Predicción Decision Tree')
# ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
# sns.kdeplot(prediction,color="g",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
# plt.legend()
# plt.grid(True)

###############################################################################
# Entrenamiento y predicción de SV Regressor
###############################################################################
print('\n <<< Predicción SVR >>>')
# model=SVR(kernel='linear',C=1001,epsilon=7,gamma=0.001)
model=SVR(kernel='linear',C=46)
start_time = pd.Timestamp.now()
model.fit(X_train, y_train)
prediction = model.predict(x_test)
mae_model=mean_absolute_error(y_test,prediction)
mse_model=np.sqrt(mean_squared_error(y_test, prediction))
r2_model=r2_score(y_test,prediction)
print('Tamaño Test: {:.0%}'.format(test_size))
print('MAE:',mae_model)
print('MSE:',mse_model)
print('R2:',r2_model)
runing_time = (pd.Timestamp.now() - start_time).total_seconds()
print('tiempo de ejecución:',runing_time)

metrica.append(['Support Vector Regressor',mae_model,mse_model,r2_model,runing_time])

plt.subplot(1,3,3)
plt.title('Predicción SVR - linear C=46')
ax1 = sns.kdeplot(y_test,color="r",label="Valor Actual",linewidth=1)
sns.kdeplot(prediction,color="c",label="Valor predicho",ax=ax1,linewidth=1,fill=True)
plt.legend()
plt.grid(True)

# Dataframe con Métricas
metrica=pd.DataFrame(data=metrica,columns=['modelo','MAE','MSE','R2','Tiempo (s)'])
print(metrica)

plt.tight_layout(pad=1)
plt.show()
