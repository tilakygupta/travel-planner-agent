from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# IBM watsonx.ai credentials (use env variables for security)
IBM_API_KEY = os.environ.get('IBM_API_KEY', 'sl_DRR_HOWRv16X0w8IqDssPuFDUsrNsGFFl6jFjosVv')
PROJECT_ID  = os.environ.get('PROJECT_ID',  '01e2efd8-fb38-41be-b5c6-cf7fb44787bc')
WX_URL      = 'https://au-syd.ml.cloud.ibm.com'

def get_iam_token():
    """Exchange IBM API Key for IAM bearer token"""
    resp = requests.post(
        'https://iam.cloud.ibm.com/identity/token',
        data={'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
              'apikey': IBM_API_KEY},
        headers={'Content-Type': 'application/x-www-form-urlencoded'}
    )
    return resp.json().get('access_token')

def ask_granite(user_message):
    """Send user message to IBM Granite via watsonx.ai"""
    token = get_iam_token()
    system_prompt = """You are an intelligent AI Travel Planner Agent powered by IBM Granite.
Help users plan trips by suggesting destinations, building itineraries,
recommending transport and accommodation based on budget and preferences.
Always ask for destination, dates, budget, number of travelers if not provided."""

    payload = {
        "model_id": "ibm/granite-3-2-8b-instruct",
        "input": f"{system_prompt}\n\nUser: {user_message}\nAssistant:",
        "parameters": {
            "max_new_tokens": 800,
            "temperature": 0.7,
            "repetition_penalty": 1.1
        },
        "project_id": PROJECT_ID
    }
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    resp = requests.post(
        f'{WX_URL}/ml/v1/text/generation?version=2023-05-29',
        json=payload, headers=headers
    )
    result = resp.json()
    return result['results'][0]['generated_text'].strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '')
    if not user_msg:
        return jsonify({'error': 'No message provided'}), 400
    response = ask_granite(user_msg)
    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)