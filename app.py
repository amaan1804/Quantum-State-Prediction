from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Constants
L = 1.0  # Length
m = 1.0  # Mass of the particle
hbar = 1.0545718176461565e-34

# Load the trained model for predicting quantum state
model = joblib.load('rf_reg.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            x = float(request.form['x'])

            # Calculate wavefunction, eigen energy, and probability density
            wavefunction = np.sqrt(2 / L) * np.sin(np.pi * x / L)
            prob_density = np.abs(wavefunction) ** 2
            eigen_energy = ((np.pi ** 2) * (hbar ** 2)) / (2 * m * L ** 2)

            # Make predictions using the trained model
            features = [[x, wavefunction, prob_density, eigen_energy]]
            predicted_state = model.predict(features)

            # Convert the predicted state to an integer
            predicted_state_int = int(predicted_state[0])

            return render_template('result.html', x=x, predicted_state=predicted_state_int, wavefunction=wavefunction,
                                   eigen_energy=eigen_energy, prob_density=prob_density)
        except Exception as e:
            print(e)
            return render_template('index.html', error=True)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
