import re
from datetime import datetime
import comandosNumpy,Activity4LinearRegression
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"


@app.route("/hello/")
@app.route("/hello/<name>")
def hello_there(name = None):
    return render_template(
        "hellothere.html",
        name=name,
        date=datetime.now()
    )

@app.route("/numpy/")
def numpy():
    datos_html = comandosNumpy.generar_datos_numpy()  
    return render_template("numpy.html", datos=datos_html)

@app.route("/linearRegression/", methods=["GET", "POST"])
def linearRegressionExample():
    resultado = None
    
    if request.method == "POST":
        horas = float(request.form["horas"])
        resultado = Activity4LinearRegression.predecir_nota(horas)

    grafico = Activity4LinearRegression.generar_grafico()
    return render_template("LinearRegression.html", imagen=grafico, resultado=resultado)

@app.route('/LogisticRegression/')
def LogisticRegression():
    return render_template('LogisticRegressionUseCase.html')
###
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
acc = data['accuracy']
report = data['report']
cm_plot = data['cm_plot']

@app.route('/LogisticRegressionRetail/', methods=['GET', 'POST'])
def LogisticRegressionRetail():
    prediction = None
    ingresos_val = zona_val = tamano_val = visitas_val = None
    error = None
    if request.method == 'POST':
        try:
            ingresos_val = float(request.form['ingresos'])
            zona_val = float(request.form['zona'])
            tamano_val = float(request.form['tamano'])
            visitas_val = float(request.form['visitas'])
            
            X_new = np.array([[ingresos_val, zona_val, tamano_val, visitas_val]])
            pred = model.predict(X_new)[0]
            prediction = 'Compra la propiedad (1)' if pred == 1 else 'No compra la propiedad (0)'
        except ValueError:
            error = "Por favor ingresa valores numéricos válidos."
            
    return render_template('index.html',
                           accuracy=acc,
                           report=report,
                           confusion_matrix_plot=cm_plot,
                           prediction=prediction,
                           ingresos_val=ingresos_val,
                           zona_val=zona_val,
                           tamano_val=tamano_val,
                           visitas_val=visitas_val,
                           error=error)

