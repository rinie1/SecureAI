import socket
import numpy as np
from PIL import Image
import tenseal as ts
import argparse
import json
import subprocess

def recv_exact(sock, n):
    """
    Receive exactly n bytes from the socket.
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Connection closed unexpectedly.")
        data += packet
    return data

def load_model_parameters(name):
    """
    Load model parameters (weights and biases) from a .json file.
    """
    with open(name, "r") as f:
        params = json.load(f)

    W1 = np.array(params["W1"])
    b1 = np.array(params["b1"])
    W2 = np.array(params["W2"])
    b2 = np.array(params["b2"])
    return W1, b1, W2, b2

def preprocess_image(image_path):
    """
    Load and preprocess the image:
    - Convert to grayscale
    - Resize to 28x28
    - Normalize pixel values to [-1, 1]
    - Flatten to a 1D array of size 784
    """
    img = Image.open(image_path).convert('L').resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # Normalize to [-1, 1]
    return arr.flatten()

def compute_hidden_layer(W1, b1, input_vector):
    """
    Compute the hidden layer activations using ReLU activation function.
    """
    hidden = np.dot(W1, input_vector) + b1
    return np.maximum(hidden, 0.0)  # Apply ReLU

def generate_mask_and_encrypted_vector(hidden, W2, context):
    """
    Generate a random mask, compute W2·r for later unmasking,
    and encrypt the masked hidden vector.
    """
    r = np.random.randn(hidden.shape[0])  # Random mask
    Wr = W2.dot(r)  # Compute W2·r for unmasking
    hidden_masked = hidden + r  # Apply mask
    enc_hidden = ts.ckks_vector(context, hidden_masked.tolist())  # Encrypt
    return r, Wr, enc_hidden

def setup_tenseal_context():
    """
    Initialize and configure the TenSEAL context for CKKS scheme.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 20, 40]
    )
    context.global_scale = 2**20
    context.generate_galois_keys()
    return context

def send_data_to_server(host, port, context, enc_hidden):
    """
    Send the TenSEAL context and encrypted hidden vector to the server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))

        # Serialize and send context
        ctx_bytes = context.serialize(save_secret_key=False)
        sock.sendall(len(ctx_bytes).to_bytes(4, 'big'))
        sock.sendall(ctx_bytes)

        # Serialize and send encrypted hidden vector
        enc_bytes = enc_hidden.serialize()
        sock.sendall(len(enc_bytes).to_bytes(4, 'big'))
        sock.sendall(enc_bytes)

        # Receive encrypted logits from server
        enc_logits = []
        for _ in range(10):
            length = int.from_bytes(recv_exact(sock, 4), 'big')
            data = recv_exact(sock, length)
            enc_score = ts.ckks_vector_from(context, data)
            enc_logits.append(enc_score)

    return enc_logits

def decrypt_and_unmask_logits(enc_logits, Wr):
    """
    Decrypt the received logits and remove the effect of the mask.
    """
    logits = np.array([score.decrypt()[0] for score in enc_logits])
    logits_unmasked = logits - Wr  # Remove mask
    return logits_unmasked

def predict_class(logits):
    """
    Determine the predicted class from the logits.
    """
    return int(np.argmax(logits))

def main():
    # Paths and server configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image",
                        type=str,
                        help="path/to/image.jpg\nIf you don't have image don't use this flag\nAfter that you can draw your digit by yourself",
                        default=None)
    parser.add_argument("-m", "--model",
                        type=str,
                        help="Json model data W1, b1, W2, b2. Default 'model_params.json'",
                        default="model_params.json")
    parser.add_argument("--host",
                        type=str,
                        help="hostname",
                        default="127.0.0.1")
    parser.add_argument("-p", "--port",
                        type=int,
                        help="port",
                        default=9000)
    args = parser.parse_args()

    if args.image == None:
        subprocess.run(['python', 'holst.py'])
        args.image = "digit.jpg"

    model_path = args.model
    image_path = args.image
    HOST, PORT = args.host, args.port

    # Load model parameters
    W1, b1, W2, b2 = load_model_parameters(model_path)

    # Preprocess input image
    input_vector = preprocess_image(image_path)

    # Compute hidden layer activations
    hidden = compute_hidden_layer(W1, b1, input_vector)

    # Setup TenSEAL context
    context = setup_tenseal_context()

    # Generate mask and encrypt hidden vector
    r, Wr, enc_hidden = generate_mask_and_encrypted_vector(hidden, W2, context)

    # Send data to server and receive encrypted logits
    enc_logits = send_data_to_server(HOST, PORT, context, enc_hidden)

    # Decrypt logits and remove mask
    logits_unmasked = decrypt_and_unmask_logits(enc_logits, Wr)

    # Predict class
    predicted_class = predict_class(logits_unmasked)
    print("Predicted class:", predicted_class)

if __name__ == "__main__":
    main()
