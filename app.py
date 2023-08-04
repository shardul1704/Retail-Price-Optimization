from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    product_id = request.form['product_id']
    total_price = float(request.form['total_price'])
    freight_price = float(request.form['freight_price'])
    customers = int(request.form['customers'])
    comp1_diff = float(request.form['comp1_diff'])
    comp2_diff = float(request.form['comp2_diff'])
    comp3_diff = float(request.form['comp3_diff'])
    fp1_diff = float(request.form['fp1_diff'])
    fp2_diff = float(request.form['fp2_diff'])
    fp3_diff = float(request.form['fp3_diff'])
    product_score = float(request.form['product_score'])

    # Prepare the input data for prediction
    input_data = [[
        total_price, freight_price, customers, comp1_diff, comp2_diff,
        comp3_diff, fp1_diff, fp2_diff, fp3_diff, product_score
    ]]

    # Perform the unit price prediction
    unit_price_pred = model.predict(input_data)[0]

    return f"Predicted Unit Price for Product ID {product_id}: {unit_price_pred}"



if __name__ == '__main__':
    app.run()


