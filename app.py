import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    ip_features = [x for x in request.form.values()]
    features = [np.array(ip_features)]
    scaler=StandardScaler()
    scaled_features=scaler.fit_transform(features)
    prediction = model.predict(scaled_features)

    if prediction[0]==1:
        return render_template('result.html', prediction_text="The Patient has Diabetes !!")
    return render_template('result.html', prediction_text="The Patient Does not have Diabetes !!")

if __name__ == "__main__":
    app.run(debug=True)
