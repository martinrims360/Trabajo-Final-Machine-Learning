# 📊 Reconocimiento de Dígitos Manuscritos

## 📌 Descripción del proyecto

Este proyecto tiene como objetivo desarrollar un sistema capaz de reconocer dígitos escritos a mano utilizando técnicas de Machine Learning y visión computacional.

Se emplea una Red Neuronal Convolucional (CNN) como modelo principal, debido a su alto rendimiento en el procesamiento de imágenes. Además, se realiza una comparación con otros algoritmos como KNN y SVM.

---

## 🎯 Objetivo

* Clasificar correctamente dígitos manuscritos (0–9)
* Comparar el rendimiento de distintos modelos
* Evaluar la precisión mediante métricas de clasificación

---

## 📂 Dataset

Se utilizó el dataset **MNIST**, el cual contiene:

* 70,000 imágenes de dígitos manuscritos
* Imágenes en escala de grises de 28x28 píxeles
* Clases del 0 al 9

---

## ⚙️ Tecnologías utilizadas

* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy
* Matplotlib
* OpenCV

---

## 🔄 Preprocesamiento

Antes del entrenamiento, se realizaron los siguientes pasos:

* Normalización de imágenes (valores entre 0 y 1)
* Redimensionamiento a 28x28 píxeles
* Reestructuración de datos para la red neuronal
* Conversión a formato adecuado para modelos tradicionales

---

## 🧠 Modelos implementados

### 🔷 CNN (Modelo principal)

* Capas convolucionales para extracción de características
* Capas de pooling para reducir dimensionalidad
* Capas densas para clasificación
* Función de activación: ReLU y Softmax
* Optimizador: Adam

### 🔷 KNN

* Clasificación basada en vecinos más cercanos

### 🔷 SVM

* Clasificación mediante separación de hiperplanos

---

## 📊 Evaluación del modelo

Se utilizaron las siguientes métricas:

* Accuracy
* Precision
* Recall
* F1-score
* Matriz de confusión

---

## 📈 Resultados

* CNN: ~98% – 99% de precisión
* SVM: ~96% – 97%
* KNN: ~94% – 96%

La matriz de confusión mostró valores altos en la diagonal principal, indicando una correcta clasificación en la mayoría de los casos.

---

## 🖼️ Resultados visuales

El modelo fue capaz de predecir correctamente los dígitos.
Por ejemplo, se obtuvo la predicción correcta del número **7** a partir de una imagen manuscrita.

![Predicción del modelo](Figure_numero_7.png)

---

## ✅ Conclusiones

* La CNN obtuvo el mejor rendimiento en comparación con KNN y SVM.
* El modelo logró una alta precisión en la clasificación de dígitos manuscritos.
* El sistema es viable para automatizar el reconocimiento de datos en formularios.

---

## ▶️ Ejecución del proyecto

```bash
python -m venv venv310
venv310\Scripts\activate

pip install -r requirements.txt
python src/main.py
```

---

## 📌 Autor

Proyecto académico de Machine Learning
Reconocimiento de dígitos manuscritos
