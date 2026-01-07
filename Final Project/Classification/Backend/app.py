from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("crop_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]

        prediction = model.predict([features])[0]
        return render_template('result.html', crop=prediction)

    except:
        return render_template('result.html', crop="Error in input values")

if __name__ == '__main__':
    app.run(debug=True)
