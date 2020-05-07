import sys
sys.path.append('.')
from flask import Flask, jsonify, request, abort

from lib import custom_processor
import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return "<h1> You are home now site back and relax! </h1>"

model_path = "./lib/xgb_model.pkl"
xgb_model = joblib.load(model_path)


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST' and "data" in request.json and len(request.get_json()["data"]) > 2:
        try:
            raw = request.get_json()["data"]

            processor = custom_processor.InputTransformer()

            data = processor.transform(raw)
        except ValueError:
            app.logger.error("Request has no data or is not a valid json object")
            abort(400)

        predictions = xgb_model.predict(data)
        probability = xgb_model.predict_proba(data)
        results = dict()

        if type(raw) is list:
            for idx, item in enumerate(raw):
                results[idx] = {
                                "prediction":list(predictions)[idx],
                                "probability":list(probability)[idx][1],
                                "message": item
                                }
            return jsonify({"results": str(results)})

        elif type(raw) is str:
            results[0] = {
                          "prediction":list(predictions)[0],
                          "probability":list(probability)[0][1],
                          "message": raw
                        }
            return jsonify({"results": str(results)})
    else:
        app.logger.error("wrong data entered please enter as text")
        abort(400)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8441, debug=True)

