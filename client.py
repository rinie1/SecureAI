import socket
import numpy as np
from PIL import Image
import tenseal as ts
import argparse
import json
from holst import SmoothDigitDrawer
import tkinter as tk
import os

def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Connection closed unexpectedly.")
        data += packet
    return data

def load_model_parameters(name):
    with open(name, "r") as f:
        params = json.load(f)

    W1 = np.array(params["W1"])
    b1 = np.array(params["b1"])
    W2 = np.array(params["W2"])
    b2 = np.array(params["b2"])
    return W1, b1, W2, b2

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L').resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.flatten()

def compute_hidden_layer(W1, b1, input_vector):
    hidden = np.dot(W1, input_vector) + b1
    return np.maximum(hidden, 0.0)

def generate_mask_and_encrypted_vector(hidden, W2, context):
    r = np.random.randn(hidden.shape[0])
    Wr = W2.dot(r)
    hidden_masked = hidden + r
    enc_hidden = ts.ckks_vector(context, hidden_masked.tolist())
    return r, Wr, enc_hidden

def setup_tenseal_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 20, 40]
    )
    context.global_scale = 2**20
    context.generate_galois_keys()
    return context

def send_data_to_server(host, port, context, enc_hidden):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))

        ctx_bytes = context.serialize(save_secret_key=False)
        sock.sendall(len(ctx_bytes).to_bytes(4, 'big'))
        sock.sendall(ctx_bytes)

        enc_bytes = enc_hidden.serialize()
        sock.sendall(len(enc_bytes).to_bytes(4, 'big'))
        sock.sendall(enc_bytes)

        enc_logits = []
        for _ in range(10):
            length = int.from_bytes(recv_exact(sock, 4), 'big')
            data = recv_exact(sock, length)
            enc_score = ts.ckks_vector_from(context, data)
            enc_logits.append(enc_score)

    return enc_logits

def decrypt_and_unmask_logits(enc_logits, Wr):
    logits = np.array([score.decrypt()[0] for score in enc_logits])
    logits_unmasked = logits - Wr
    return logits_unmasked

def predict_class(logits):
    return int(np.argmax(logits))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, default=None)
    parser.add_argument("-m", "--model", type=str, default="model_params.json")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=9000)
    args = parser.parse_args()
    if args.image is None:

        root = tk.Tk()
        app = SmoothDigitDrawer(root)
        root.mainloop()

        args.image = "digit.jpg"

        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Файл {args.image} не найден. Убедитесь, что вы сохранили изображение в холсте.")

    W1, b1, W2, b2 = load_model_parameters(args.model)
    input_vector = preprocess_image(args.image)
    print("Original image:", input_vector[:10])
    hidden = compute_hidden_layer(W1, b1, input_vector)
    context = setup_tenseal_context()
    r, Wr, enc_hidden = generate_mask_and_encrypted_vector(hidden, W2, context)

    # Уведомление о шифровании данных перед отправкой
    print("Client: Encrypting and sending data to server (user data remains encrypted).")

    enc_logits = send_data_to_server(args.host, args.port, context, enc_hidden)
    logits_unmasked = decrypt_and_unmask_logits(enc_logits, Wr)
    predicted_class = predict_class(logits_unmasked)
    print("Predicted class:", predicted_class)

if __name__ == "__main__":
    main()
