import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

# Datos
data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Entrenar modelo de regresión lineal
X = df[["Study Hours"]]
y = df["Final Grade"]
model = LinearRegression()
model.fit(X, y)

# Función para predecir y generar la imagen del gráfico
def generar_grafico():
    plt.figure(figsize=(6,4))
    
    # Scatter plot de los datos
    plt.scatter(df["Study Hours"], df["Final Grade"], color="blue", label="Datos Reales")
    
    # Línea de regresión
    x_range = np.linspace(min(X.values), max(X.values), 100).reshape(-1,1)
    y_pred = model.predict(x_range)
    plt.plot(x_range, y_pred, color="red", label="Regresión Lineal")
    
    plt.xlabel("Horas de Estudio por Semana")
    plt.ylabel("Nota Final (0-5)")
    plt.title("Regresión Lineal: Horas de Estudio vs Nota Final")
    plt.legend()
    
    # Guardar la imagen en memoria
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    
    # Convertir a base64 para mostrar en Flask
    img_base64 = base64.b64encode(img.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

# Función para predecir una nueva nota
def predecir_nota(horas):
    prediccion = model.predict([[horas]])[0]
    return round(prediccion, 2)
