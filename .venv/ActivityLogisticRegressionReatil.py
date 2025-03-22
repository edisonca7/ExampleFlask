import numpy as np
import matplotlib
matplotlib.use('Agg')  # Para entornos sin display
import matplotlib.pyplot as plt
import io
import base64
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def create_confusion_matrix_figure(cm):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()
    plt.xticks([0, 1], ['No compra (0)', 'Compra (1)'])
    plt.yticks([0, 1], ['No compra (0)', 'Compra (1)'])
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')

    # Etiquetar las celdas
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return encoded

def train_and_save_model():
    # =========================
    # 1. Cargar datos controlados
    # =========================
    # Se definen arrays fijos para cada variable independiente
    Ingresos = np.array([3000, 4500, 6000, 8000, 10000, 2500, 4000, 5500, 7000, 8500, 9000, 3200, 4800, 6400, 7200])
    zona     = np.array([1,    2,    1,    3,    4,     0,    2,    3,    1,    2,    4,    1,    2,    3,    0])
    tamano   = np.array([80,   100,  90,   120,  150,   70,   110,  130,  85,   95,   140,  75,   105,  115,  65])
    visitas  = np.array([2,    4,    3,    6,    8,     1,    5,    7,    3,    4,    9,    2,    4,    6,    1])

    # Construir la matriz de features
    X = np.column_stack((Ingresos, zona, tamano, visitas))
    
    # =========================
    # 2. Calcular la variable dependiente
    # =========================
    # Se utiliza una función logística con coeficientes elegidos para que tenga sentido:
    p = 1 / (1 + np.exp(-(0.0002 * Ingresos + 0.5 * zona + 0.03 * tamano + 0.3 * visitas - 5)))
    # Se define la compra: 1 si la probabilidad es mayor o igual a 0.5, 0 en otro caso.
    compra = (p >= 0.5).astype(int)
    y = compra

    # =========================
    # 3. Separar datos en entrenamiento y prueba
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # =========================
    # 4. Entrenar el modelo
    # =========================
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # =========================
    # 5. Evaluar el modelo
    # =========================
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm_plot = create_confusion_matrix_figure(cm)

    # =========================
    # 6. Guardar modelo y métricas en un diccionario
    # =========================
    data = {
        'model': model,
        'accuracy': acc,
        'report': report,
        'cm_plot': cm_plot
    }
    with open('model.pkl', 'wb') as f:
        pickle.dump(data, f)

    print("Modelo y métricas guardados en 'model.pkl'")

if __name__ == '__main__':
    train_and_save_model()
