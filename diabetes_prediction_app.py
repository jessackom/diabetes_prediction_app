# app.py
import json
import sys
from typing import Dict, Any, Union

import requests
import pandas as pd
from flask import Flask, render_template, request, jsonify

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Print config at startup
Config.print_config_status()

def create_tf_serving_json(data: Union[Dict, pd.DataFrame]) -> Dict:
    if isinstance(data, dict):
        # make values lists (single-row -> list of 1)
        return {'inputs': {name: [data[name]] for name in data.keys()}}
    # For DataFrame, return dataframe_split for MLflow-style input
    return {'dataframe_split': data.to_dict(orient='split')}

def validate_input_data(data: Dict) -> tuple[bool, str]:
    required_features = Config.MODEL_FEATURES
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        return False, f"Missing required features: {', '.join(missing_features)}"
    for feature in required_features:
        try:
            float(data[feature])
        except (ValueError, TypeError):
            return False, f"Invalid value for feature '{feature}': must be a number"
    return True, ""

def score_model(dataset: pd.DataFrame) -> Dict[str, Any]:
    url = Config.MLFLOW_ENDPOINT_URL
    token = Config.DATABRICKS_TOKEN
    timeout = Config.REQUEST_TIMEOUT

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    # MLflow typically accepts dataframe_split for pandas DataFrame
    data_dict = {'dataframe_split': dataset.to_dict(orient='split')}
    data_json = json.dumps(data_dict, allow_nan=True)

    try:
        response = requests.post(url=url, headers=headers, data=data_json, timeout=timeout)
    except requests.exceptions.Timeout:
        raise Exception(f'Request timed out after {timeout} seconds. The model endpoint may be slow or unavailable.')
    except requests.exceptions.ConnectionError:
        raise Exception('Failed to connect to the model endpoint. Check your internet connection and endpoint URL.')

    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}. Response: {response.text}')

    return response.json()

def build_dataframe_from_request(data: Dict) -> pd.DataFrame:
    feature_dict = {
        feature: float(data.get(feature, Config.DEFAULT_VALUES.get(feature, 0.0)))
        for feature in Config.MODEL_FEATURES
    }
    return pd.DataFrame([feature_dict])

@app.route('/')
def home():
    # Renders templates/index.html — simple form to enter features
    return render_template('index.html', features=Config.MODEL_FEATURES, defaults=Config.DEFAULT_VALUES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided in request body'}), 400

        is_valid, error_message = validate_input_data(data)
        if not is_valid:
            return jsonify({'success': False, 'error': error_message}), 400

        df = build_dataframe_from_request(data)
        result = score_model(df)

        # MLflow / Databricks serving responses vary. Try common extraction patterns.
        if isinstance(result, dict) and 'predictions' in result:
            prediction_value = result['predictions'][0]
        elif isinstance(result, dict) and 'data' in result:
            # Some endpoints return {"data": [[prob, ...]]}
            prediction_value = result['data'][0]
        else:
            prediction_value = result

        return jsonify({'success': True, 'prediction': prediction_value})

    except requests.exceptions.RequestException as e:
        return jsonify({'success': False, 'error': f'Network error while calling model endpoint: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    is_valid, errors = Config.validate_config()
    if is_valid:
        return jsonify({'status': 'healthy', 'app': Config.APP_NAME, 'version': Config.APP_VERSION}), 200
    else:
        return jsonify({'status': 'unhealthy', 'errors': errors}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("STARTING MANUFACTURING DEFECT PREDICTION APP")
    print("="*70)
    is_valid, errors = Config.validate_config()
    if not is_valid:
        print("\n❌ CONFIGURATION ERROR!\n")
        for error in errors:
            print(f"  • {error}")
        print("\nSet the required environment variables (see README).")
        sys.exit(1)

    print("\n✅ Configuration validated successfully!\n")
    print(f"🚀 Starting server at http://{Config.HOST}:{Config.PORT}")
    print(f"📊 Model endpoint: {Config.MLFLOW_ENDPOINT_URL}")
    print("="*70 + "\n")

    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
