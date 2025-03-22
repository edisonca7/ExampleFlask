import numpy as np

def generar_datos_numpy():
    """Genera datos de NumPy y los devuelve en HTML"""
    
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([[1, 2, 3], [4, 5, 6]])

    html_data = f"""
    <h2>Arrays de NumPy</h2>
    <p><b>Array 1D:</b> {a.tolist()}</p>
    <p><b>Array 2D:</b></p>
    <table border="1">
        {''.join(f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>" for row in b)}
    </table>

    <h2>Propiedades</h2>
    <p><b>Forma:</b> {b.shape}</p>
    <p><b>Tamaño:</b> {b.size}</p>
    <p><b>Tipo de datos:</b> {b.dtype}</p>

    <h2>Operaciones Matemáticas</h2>
    <p><b>Suma:</b> {np.array([10, 20, 30, 40]) + np.array([1, 2, 3, 4])}</p>
    <p><b>Multiplicación:</b> {np.array([10, 20, 30, 40]) * np.array([1, 2, 3, 4])}</p>
    """

    return html_data
