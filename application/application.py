from enums import *
from data_transformer import *
from model import *

import pickle

from flask import Flask, render_template, request, redirect, url_for

resources_path = os.getcwd() + '/resources'
data_transformer_path = resources_path + '/data_transformer.pkl'
model_info = (resources_path, '.joblib', '/weights.json', '/scores.json')

with open(data_transformer_path, 'rb') as file:
    data_transformer = pickle.load(file)
model = Model(model_info)

app = Flask(__name__)
app.secret_key = 'your_secret_key'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/credit_application/profile', methods=['GET', 'POST'])
def credit_application_profile():
    if request.method == 'POST':
        profile = Profile(request.form)
        print("Input profile:", profile.get_dictionary())

        model_name = PredictionModel.get_name(request.form['model'])
        processed_profile = data_transformer.process_profile(profile)
        print("Processed profile:", processed_profile)

        result = model.predict(model_name, processed_profile)
        return redirect(url_for('result', result=result))
    else:
        return render_template('credit_application/client_profile.html', home_ownership_enum=HomeOwnership,
                               loan_intent_enum=LoanIntent, loan_grade_enum=LoanGrade,
                               previous_default_enum=PreviousDefault, prediction_model_enum=PredictionModel)


@app.route('/credit_application/result')
def result():
    return render_template('credit_application/prediction_result.html',
                           result=request.args.get('result', 'Something went wrong...'))


@app.route('/models_weights')
def models_weights():
    return render_template('/models_weights.html', weights=model.weights)


@app.route('/models_scores')
def models_scores():
    return render_template('/models_scores.html', scores=model.scores)


if __name__ == '__main__':
    app.run(host='localhost', port=8080)
