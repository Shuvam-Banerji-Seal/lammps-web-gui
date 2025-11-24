from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/api/data")
def get_data():
    with open("c60.data", "r") as f:
        data = f.read()
    return jsonify({"data": data})

if __name__ == "__main__":
    app.run(debug=True)
