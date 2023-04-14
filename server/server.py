"""Lifelike Toolkit Flask server"""
from flask import Flask, make_response
from controller.characterController import tellCharacter

app = Flask(__name__)

@app.route('/ping')
def ping():
    return {
        'message': 'ping'
    }

@app.route('/character/<name>/tell/<dialogue>')
def tell(name, dialogue):
    """
    Get request for a response
    """
    response = make_response(tellCharacter(name, dialogue))
    # Enable Access-Control-Allow-Origin
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response