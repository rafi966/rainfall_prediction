from flask import Flask, render_template, request
from flask_cors import cross_origin
import pandas as pd
import pickle
import logging

app = Flask(__name__, template_folder="templates")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

try:
    model = pickle.load(open("./models/cat.pkl", "rb"))
    logging.info("Model Loaded Successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        try:
            # Print the form data for debugging
            logging.debug(f"Form data: {request.form}")

            # DATE
            date = request.form.get('date', '')
            day = float(pd.to_datetime(date, format="%Y-%m-%d").day) if date else 0
            month = float(pd.to_datetime(date, format="%Y-%m-%d").month) if date else 0

            # Helper function to convert form data to float
            def to_float(value):
                return float(value) if value else 0

            # Form fields
            minTemp = to_float(request.form.get('mintemp', ''))
            maxTemp = to_float(request.form.get('maxtemp', ''))
            rainfall = to_float(request.form.get('rainfall', ''))
            evaporation = to_float(request.form.get('evaporation', ''))
            sunshine = to_float(request.form.get('sunshine', ''))
            windGustSpeed = to_float(request.form.get('windgustspeed', ''))
            windSpeed9am = to_float(request.form.get('windspeed9am', ''))
            windSpeed3pm = to_float(request.form.get('windspeed3pm', ''))
            humidity9am = to_float(request.form.get('humidity9am', ''))
            humidity3pm = to_float(request.form.get('humidity3pm', ''))
            pressure9am = to_float(request.form.get('pressure9am', ''))
            pressure3pm = to_float(request.form.get('pressure3pm', ''))
            temp9am = to_float(request.form.get('temp9am', ''))
            temp3pm = to_float(request.form.get('temp3pm', ''))
            cloud9am = to_float(request.form.get('cloud9am', ''))
            cloud3pm = to_float(request.form.get('cloud3pm', ''))
            location = to_float(request.form.get('location', ''))
            winddDir9am = to_float(request.form.get('winddir9am', ''))
            winddDir3pm = to_float(request.form.get('winddir3pm', ''))
            windGustDir = to_float(request.form.get('windgustdir', ''))
            rainToday = to_float(request.form.get('raintoday', ''))

            input_lst = [
                location, minTemp, maxTemp, rainfall, evaporation, sunshine,
                windGustDir, windGustSpeed, winddDir9am, winddDir3pm, windSpeed9am, windSpeed3pm,
                humidity9am, humidity3pm, pressure9am, pressure3pm, cloud9am, cloud3pm, temp9am, temp3pm,
                rainToday, month, day
            ]

            logging.debug(f"Input list: {input_lst}")

            if model:
                pred = model.predict([input_lst])[0]
                logging.debug(f"Prediction: {pred}")
                output = pred
                if output == 0:
                    return render_template("after_sunny.html")
                else:
                    return render_template("after_rainy.html")
            else:
                logging.error("Model is not loaded")
                return "Model not loaded", 500
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return f"Error during prediction: {e}", 500
    return render_template("predictor.html")

if __name__ == '__main__':
    app.run(debug=True)
