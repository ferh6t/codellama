from flask import Flask, request, jsonify
from typing import Optional
import fire
from llama import Llama
import argparse

app = Flask(__name__)
max_gen_len: int = 4096
temperature: float = 0.2
top_p: float = 0.9

def askChatModel(data):
    dialogs: List[Dialog] = [
        [{"role": "user", "content": data}]        
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    print(results)
    # for prompt, result in zip(prompts, results):
    #     print(prompt)
    #     print(f"> {result['generation']}")
    #     print("\n==================================\n")
    
    return results[0]['generation']


# Define a route to accept POST requests with input data
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Extract input data from the request
        data = request.get_json()
        print(data['question'])
        # Call your PyTorch method here
        result = askChatModel(data['question'])
        print(result)
        # Format the result and return as JSON
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})



def createChatGenerator(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 4096,
    max_batch_size: int = 64,
):
    global generator
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    
# Define your PyTorch method
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Sample Server Script")

    # Define command-line arguments
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer path")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--max_batch_size", type=int, default=4, help="Max batch size")

    args = parser.parse_args()

    createChatGenerator(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )
    app.run( port=5001 )

