import socket
import numpy as np
import tenseal as ts
import argparse

def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Connection closed unexpectedly.")
        data += packet
    return data

def load_model_parameters(model_path):
    data = np.load(model_path)
    W2 = data['W2']
    b2 = data['b2']
    return W2, b2

def receive_context_and_encrypted_vector(conn):
    ctx_size = int.from_bytes(recv_exact(conn, 4), 'big')
    ctx_bytes = recv_exact(conn, ctx_size)
    context = ts.context_from(ctx_bytes)

    vec_size = int.from_bytes(recv_exact(conn, 4), 'big')
    enc_hidden = ts.ckks_vector_from(context, recv_exact(conn, vec_size))

    return context, enc_hidden

def compute_encrypted_logits(enc_hidden, W2, b2):
    # Все вычисления выполняются над зашифрованными данными; сервер не видит содержимого enc_hidden
    enc_logits = []
    for i in range(10):
        score = enc_hidden.dot(W2[i].tolist()) + float(b2[i])
        enc_logits.append(score)
    return enc_logits

def send_encrypted_logits(conn, enc_logits):
    for score in enc_logits:
        data = score.serialize()
        conn.sendall(len(data).to_bytes(4, 'big') + data)

def start_server(host, port, model_path, log_file):
    W2, b2 = load_model_parameters(model_path)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, port))
        sock.listen(1)
        f = open(log_file, "w")
        print(f"Server started at {host}:{port}, waiting for connection...")
        f.write(f"Server started at {host}:{port}, waiting for connection...\n")
        f.close()

        while True:
            conn, addr = sock.accept()
            with conn:
                f = open(log_file, "a")
                print(f"Client connected from {addr}")
                f.write(f"Client connected from {addr}\n")

                context, enc_hidden = receive_context_and_encrypted_vector(conn)
                print("Server: Received encrypted data; performing computation on ciphertext (user data is not visible).")
                f.write("Server: Received encrypted data; performing computation on ciphertext (user data is not visible).\n")

                # Попытка расшифровки (демонстрация невозможности)
                try:
                    print("Enc vector:", enc_hidden.serialize()[:10])
                    f.write(f"Enc vector: {enc_hidden.serialize()[:10]}\n")
                    print("Attempting to decrypt encrypted vector on server...")
                    f.write("Attempting to decrypt encrypted vector on server...\n")
                    dec = enc_hidden.decrypt()
                    print("Decryption result (unexpected!):", dec)
                    f.write(f"Decryption result (unexpected!): {dec}\n")
                except Exception as e:
                    print("Decryption failed (as expected). Server does not have the secret key.")
                    f.write("Decryption failed (as expected). Server does not have the secret key.\n")
                    print("Error:", e)
                    f.write(f"Error: {e}\n")

                enc_logits = compute_encrypted_logits(enc_hidden, W2, b2)

                print("Enc logits:", enc_logits[0].serialize()[:10])
                f.write(f"Enc logits: {enc_logits[0].serialize()[:10]}\n")

                send_encrypted_logits(conn, enc_logits)
                print("Encrypted logits sent to client.")
                f.write("Enc logist sent to client.\n")

                close_conn = input("Close connection? (y/n) ")
                if close_conn.lower() not in ['n', 'no']:
                    break

                f.close()

        print("Closing connection.")
        f.write("Closing connection\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="mnist_model_split.npz")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("-p", "--port", type=int, default=5000)
    parser.add_argument("-f", "--file", type=str, default="logs.txt")
    args = parser.parse_args()

    start_server(args.host, args.port, args.model, args.file)