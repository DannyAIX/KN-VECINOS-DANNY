# CRISP-DM CRISP-DM significa Cross Industry Standard Process for Data Mining, y es el estándar más usado en la industria para desarrollar proyectos de anális
# KNN
#Clasificador de Vinos con KNN
#Entrena un modelo de K-Vecinos más Cercanos (KNN) para predecir la calidad de un vino tinto a partir de sus características químicas. 
## Utilizaremos un dataset de vinos tintos extraido de Wine Quality Data Set - UCI

# Paso 1: Obtener Datos de CSV

import pandas as pd

#total_data = pd.read_csv("/workspaces/machinelearning/data/raw/airbnb2019.csv")  # pasar la data a un data fram

# si la voy a traer de la pagina
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv",sep=';')
#total_data.to_csv("/workspaces/regresion-logistica/data/raw/total_data.csv", sep=';', index = False)

#Paso 2 Entender o explorar la data
print(total_data.head()) #ver rapidamente si cargo la info

print(total_data.shape)  # 12 columnas o variables y 1,599 filas o cantidad de registros
print(total_data.columns) # ver las columnas
print(total_data.info()) # ves tipos de datos, valores nulos y memoria usada, todo de un vistazo. float64(11), int64(1)

print(total_data.describe()) # ESTADISTICAS de cada columna,

#Descripción de las columnas
#Cada fila representa un vino. Las columnas describen su composición química: fixed acidity, volatile acidity, citric acid
# residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

# en el ejercicio dice que la variable objetivo es label pero no existe, 
# pero la quality tiene una media de 5.63, un minimo de 3 un maximo de 8 y una mediana de 6

#el ejercicio dice que es label (pero no existe por lo q la crearemos):

#0 = Baja calidad - quality 3 y 4 ó <=4
#1 = Calidad media - quality 5 y 6 ó >4 y <= 6
#2 = Alta calidad - 7 y 8 ó >6 y <= 8

def clasificar_calidad(valor):
    if valor <= 4:
        return 0
    elif valor <= 6:
        return 1
    else:
        return 2

total_data["label"] = total_data["quality"].apply(clasificar_calidad)

print('Columna Label 0 baja, 1 media, 2 alta calidad: ', total_data["label"].value_counts())

print(total_data.describe())
# se agrego la columna label

#Dividir en entrenamiento y prueba
from sklearn.model_selection import train_test_split

#separar variables independientes (X) y dependiente (y)
X = total_data.drop(["quality", "label"], axis=1)  # sacar las independientes u objetivo
y = total_data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Escalar las variables (muy importante en KNN) 
#escalar (con StandardScaler) significa: Restar la media de cada variable y Dividir entre su desviación estándar para q no haya problema de escala
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Entrenar el modelo KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

#Evaluar el rendimiento del modelo
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

y_pred = knn.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

#-------------------------------------------Optimizar el número de vecinos (K)
import matplotlib.pyplot as plt

accuracy = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = knn.score(X_test_scaled, y_test)
    accuracy.append(acc)

plt.plot(range(1, 21), accuracy, marker='o')
plt.xlabel('Número de Vecinos (K)')
plt.ylabel('Precisión')
plt.title('Selección del Mejor K')
plt.show()

best_k = accuracy.index(max(accuracy)) + 1
print("Mejor valor de K:", best_k)

#Reentrenar con el mejor K
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_best = knn_best.predict(X_test_scaled)

print("Precisión final:", accuracy_score(y_test, y_pred_best))
print("\nReporte final:\n", classification_report(y_test, y_pred_best))

#mejoro de 82% a 86% Significa que el modelo predice correctamente el 86% de los vinos del conjunto de prueba.
#esta fallando en predecir los de alta y baja calidad porque son muy pocos, en general el data set es pequeño

import joblib

# Guardar el modelo y el scaler
joblib.dump(knn_best, "modelo_knn_vino.pkl")
joblib.dump(scaler, "scaler_knn_vino.pkl")

print("✅ Modelo y scaler guardados correctamente.")

