from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("artifacts/model_path/model.pkl")

@app.route('/', methods = ['GET', 'POST'])
def index():
    predication = None
#SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
    if request.method == 'POST':
        sepal_length = float(request.form['SepalLengthCm'])
        sepal_Width = float(request.form['SepalWidthCm'])
        petal_length = float(request.form['PetalLengthCm'])
        petal_Width = float(request.form['PetalWidthCm'])

        input_data = np.array([[sepal_length,sepal_Width,petal_length,petal_Width]])

        predication = model.predict(input_data)[0]

    return render_template("index.html", predication = predication)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000 , debug=True)