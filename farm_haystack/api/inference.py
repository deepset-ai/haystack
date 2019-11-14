import json
from pathlib import Path

import numpy as np
from flask import request, make_response
from flask_cors import CORS
from flask_restplus import Api, Resource

from farm_haystack import Finder
from farm_haystack.database import app
from farm_haystack.reader.adaptive_model import FARMReader
from farm_haystack.retriever.tfidf import TfidfRetriever

CORS(app)
api = Api(
    app, debug=True, validate=True, version="1.0", title="FARM Question Answering API"
)

MODELS_DIRS = ["saved_models"]

model_paths = []
for model_dir in MODELS_DIRS:
    path = Path(model_dir)
    if path.is_dir():
        models = [f for f in path.iterdir() if f.is_dir()]
        model_paths.extend(models)

retriever = TfidfRetriever()
FINDERS = {}
for idx, model_dir in enumerate(model_paths):
    reader = FARMReader(model_dir=str(model_dir), batch_size=16)
    FINDERS[idx + 1] = Finder(reader, retriever)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


@api.representation("application/json")
def resp_json(data, code, headers=None):
    resp = make_response(json.dumps(data, cls=NumpyEncoder), code)
    resp.headers.extend(headers or {})
    return resp


@api.route("/finders/<int:finder_id>/ask")
class InferenceEndpoint(Resource):
    def post(self, finder_id):
        finder = FINDERS.get(finder_id, None)
        if not finder:
            return "Model not found", 404

        request_body = request.get_json()
        questions = request_body.get("questions", None)
        if not questions:
            return "The request is missing 'questions' field", 400

        filters = request_body.get("filters", None)

        results = finder.get_answers(
            question=request_body["questions"][0], top_k_reader=3, filters=filters
        )

        return results


if __name__ == "__main__":
    app.run(host="0.0.0.0")
