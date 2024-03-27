from enums import *


def map_prediction_model(prediction_model: str) -> str:
    model_mapping = {
        str(PredictionModel.LOGISTIC_REGRESSION.name): str(PredictionModel.LOGISTIC_REGRESSION.name).lower(),
        str(PredictionModel.DECISION_TREE.name): str(PredictionModel.DECISION_TREE.name).lower(),
        str(PredictionModel.RANDOM_FOREST.name): str(PredictionModel.RANDOM_FOREST.name).lower(),
        str(PredictionModel.GRADIENT_BOOSTING_MACHINES.name): str(
            PredictionModel.GRADIENT_BOOSTING_MACHINES.name).lower(),
    }
    return model_mapping.get(prediction_model, "Unknown model")


class Profile:

    def __init__(self, form):
        self.age = int(form['age'])
        self.month_income = int(form['month_income'])
        self.employment_length = int(form['employment_length'])
        self.home_ownership = form['home_ownership']
        self.loan_intent = form['loan_intent']
        self.loan_amount = int(form['loan_amount'])
        self.loan_interest_rate = int(form['loan_interest_rate'])
        self.previous_loans = int(form['previous_loans'])
        self.previous_default = form['previous_default']
        self.loan_grade = form['loan_grade']

    def get_dictionary(self):
        dictionary = {
            'age': self.age,
            'month_income': self.month_income,
            'employment_length': self.employment_length,
            'home_ownership': self.home_ownership,
            'loan_intent': self.loan_intent,
            'loan_amount': self.loan_amount,
            'loan_interest_rate': self.loan_interest_rate,
            'previous_loans': self.previous_loans,
            'previous_default': self.previous_default,
            'loan_grade': self.loan_grade
        }
        return dictionary
