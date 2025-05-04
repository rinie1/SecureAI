import socket
import numpy as np
import tenseal as ts

def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Connection closed")
        data += packet
    return data

# Загрузка второго слоя (серверного)
data = np.load("mnist_model_split.npz")
W2 = data['W2']  # (10, 128)
b2 = data['b2']  # (10,)

HOST = '127.0.0.1'
PORT = 9000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("Сервер запущен...")

conn, addr = server_socket.accept()
print("Клиент подключился:", addr)

# Получаем публичный контекст
ctx_size = int.from_bytes(recv_exact(conn, 4), byteorder='big')
ctx_bytes = recv_exact(conn, ctx_size)
public_ctx = ts.context_from(ctx_bytes)

# Получаем зашифрованный скрытый вектор (128)
vec_size = int.from_bytes(recv_exact(conn, 4), byteorder='big')
enc_hidden_bytes = recv_exact(conn, vec_size)
enc_hidden = ts.ckks_vector_from(public_ctx, enc_hidden_bytes)

# Применяем W2 * x + b2
enc_logits = []
for i in range(W2.shape[0]):
    score = enc_hidden.dot(W2[i].tolist()) + float(b2[i])
    enc_logits.append(score)

# Отправляем логиты обратно
for score in enc_logits:
    data = score.serialize()
    conn.sendall(len(data).to_bytes(4, byteorder='big'))
    conn.sendall(data)

conn.close()
print("Сервер завершил работу.")
