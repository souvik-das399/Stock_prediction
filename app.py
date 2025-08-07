from flask import Flask, request, jsonify, render_template
from predictor import predict_stock_price 

app = Flask(__name__, static_folder="static", template_folder="static")

@app.route('/')
def home():
    return render_template("index.html")  

@app.route('/predict')
def predict():
    company = request.args.get("company")
    if not company:
        return jsonify({"error": "Missing 'company' parameter."})
    
    result = predict_stock_price(company)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)