from typing import Tuple

import os
import json

import joblib
import pandas as pd


class Model:
    def __init__(self, model_info: Tuple[str, str, str, str]):
        self.models, self.weights, self.scores = Model.load_prediction_models(model_info)

    def predict(self, prediction_model: str, profile_dict: dict) -> str:
        profile = pd.DataFrame.from_dict(profile_dict, orient='index').T
        prediction_result = self.models[prediction_model].predict(profile)

        if prediction_result == 1:
            result = "Accepted"
        else:
            result = "Denied"

        return result

    @staticmethod
    def load_prediction_models(model_info: Tuple[str, str, str, str]) -> Tuple[dict, dict, dict]:
        resources_path, models_extension, weights_path, scores_path = model_info

        models = {}

        for file_name in os.listdir(resources_path):
            if file_name.endswith(models_extension):
                file_path = os.path.join(resources_path, file_name)

                loaded_model = joblib.load(file_path)

                key_without_extension = os.path.splitext(file_name)[0]
                models[key_without_extension] = loaded_model

        with open(resources_path + weights_path, 'r') as file:
            weights = json.load(file)

        with open(resources_path + scores_path, 'r') as file:
            scores = json.load(file)

        print("Models:", models)
        print("Weights:", weights)
        print("Scores:", scores)

        return models, weights, scores
