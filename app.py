from flask import Flask, render_template, request
import numpy as np
import pickle


model = pickle.load(open(r'rf_reg.pkl','rb'))

app = Flask(__name__)

@app.route("/", methods=['POST','GET'])
def home():
    if request.method=='POST':
        input_dict = request.form.to_dict()
        input_values = input_dict.values()
        input_values = list(map(float, list(input_values)))
        input_values = np.array(input_values)
        input_values = input_values.reshape(1,-1)
        prediction = model.predict(input_values)[0]
        return "THE PREDICTED PRICE IS "+str("{:.2f}".format((prediction)))+"K dollars($)"
    return render_template("index1.html")


if __name__=="__main__":
    app.run(debug=True)
