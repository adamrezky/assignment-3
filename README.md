# assignment-3
// First, we import modules, and the two lines above should almost always exist on the top of the code to initiate the flask app and API.
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json

app = Flask(__name__)
api = Api(app)

# Create parser for the payload data
parser = reqparse.RequestParser()
parser.add_argument('data')

# Define how the api will respond to the post requests
class IrisClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        X = np.array(json.loads(args['data']))
        prediction = model.predict(X)
        return jsonify(prediction.tolist())

api.add_resource(IrisClassifier, '/iris')

if __name__ == '__main__':
    # Load model
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
// In the final part of the code in api.py , we load the model saved from last section so that the app knows where to get the model if any prediction is requested. Then we run the flask app in a debug mode, where it just allows arbitrary code to be executed directly in the browser if any error happens.
    app.run(debug=True)
