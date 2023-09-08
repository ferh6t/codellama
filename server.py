from flask import Flask, request, jsonify
from typing import Optional
import fire
from llama import Llama

app = Flask(__name__)

generator = 

# Define a route to accept POST requests with input data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the request
        data = request.get_json()

        # Call your PyTorch method here
        result = askModel(data['question'])

        # Format the result and return as JSON
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})



def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):

    generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
    app.run(debug=True, port=3000)

# Define your PyTorch method
def askModel(data):
        prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
#         """\
# import socket

# def ping_exponential_backoff(host: str):""",
        data
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")
        return print(f"> {result['generation']}")

    