# Importamos las bibliotecas necesarias
import joblib
import numpy as np

# Importamos Flask para crear la aplicación web
from flask import Flask
# Importamos jsonify para convertir el objeto de respuesta a un formato JSON
from flask import jsonify

# Inicializamos la aplicación de Flask
app = Flask(__name__)

# Definimos una ruta '/predict' para la API, que aceptará solicitudes GET
@app.route('/predict', methods=['GET'])
def predict():
    # Aquí estamos predeterminando los valores de X_test. En una aplicación real, estos serían los valores que el cliente enviaría a la API
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    # Hacemos una predicción con el modelo y guardamos el resultado en la variable prediction
    prediction = model.predict(X_test.reshape(1,-1))
    # Convertimos la predicción a formato JSON y la devolvemos como respuesta de la API
    return jsonify({'predictions': list(prediction)})

# Esta es la sentencia que se ejecuta cuando se corre el script
if __name__ == '__main__':
    # Cargamos el modelo desde el archivo 'best_model.pkl'
    model = joblib.load('./models/best_model.pkl')
    # Corremos la aplicación en el puerto 8080
    app.run(port=8080)
