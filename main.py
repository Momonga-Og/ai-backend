from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Specify the model explicitly
question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "")
    context = "Your predefined context here."
    answer = question_answerer(question=query, context=context)
    return jsonify(answer=answer['answer'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
