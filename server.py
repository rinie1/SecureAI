import socket
import numpy as np
import tenseal as ts
import argparse

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

def load_model_parameters(model_path):
    """
    Load model parameters (weights and biases) from a .npz file.
    """
    data = np.load(model_path)
    W2 = data['W2']  # Weights for the second layer (shape: 10 x 128)
    b2 = data['b2']  # Biases for the second layer (shape: 10,)
    return W2, b2

def receive_context_and_encrypted_vector(conn):
    """
    Receive the TenSEAL context and the encrypted hidden vector from the client.
    """
    # Receive and deserialize the TenSEAL context
    ctx_size = int.from_bytes(recv_exact(conn, 4), 'big')
    ctx_bytes = recv_exact(conn, ctx_size)
    context = ts.context_from(ctx_bytes)

    # Receive and deserialize the encrypted hidden vector
    vec_size = int.from_bytes(recv_exact(conn, 4), 'big')
    enc_hidden = ts.ckks_vector_from(context, recv_exact(conn, vec_size))

    return context, enc_hidden

def compute_encrypted_logits(enc_hidden, W2, b2):
    """
    Compute the encrypted logits by performing dot product between
    the encrypted hidden vector and each row of W2, then adding the bias.
    """
    enc_logits = []
    for i in range(10):
        # Compute dot product and add bias
        score = enc_hidden.dot(W2[i].tolist()) + float(b2[i])
        enc_logits.append(score)
    return enc_logits

def send_encrypted_logits(conn, enc_logits):
    """
    Serialize and send each encrypted logit to the client.
    """
    for score in enc_logits:
        data = score.serialize()
        conn.sendall(len(data).to_bytes(4, 'big') + data)

def start_server(host, port, model_path):
    """
    Start the server to receive encrypted data, perform computation,
    and send back the encrypted results.
    """
    # Load model parameters
    W2, b2 = load_model_parameters(model_path)

    # Set up the server socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, port))
        sock.listen(1)
        print(f"Server started at {host}:{port}, waiting for connection...")

        while True:
            conn, addr = sock.accept()
            with conn:
                print(f"Client connected from {addr}")

                # Receive context and encrypted hidden vector
                context, enc_hidden = receive_context_and_encrypted_vector(conn)

                # Compute encrypted logits
                enc_logits = compute_encrypted_logits(enc_hidden, W2, b2)

                # Send encrypted logits back to the client
                send_encrypted_logits(conn, enc_logits)
                print("Encrypted logits sent to client.")

                close_conn = input("Close connection? (y/n) ")

                if close_conn == 'yes' or close_conn == 'y':
                    break

        print("Closing connection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        type=str,
                        help="npz model data W1, b1, W2, b2. Default 'mnist_model_split.npz'",
                        default="mnist_model_split.npz")
    parser.add_argument("--host",
                        type=str,
                        help="hostname",
                        default="127.0.0.1")
    parser.add_argument("-p", "--port",
                        type=int,
                        help="port",
                        default=9000)
    args = parser.parse_args()

    HOST = args.host
    PORT = args.port
    MODEL_PATH = args.model
    start_server(HOST, PORT, MODEL_PATH)
