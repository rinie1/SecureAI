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

# Загрузка второго слоя
data = np.load("mnist_model_split.npz")
W2 = data['W2']  # (10, 128)
b2 = data['b2']  # (10,)

HOST, PORT = '127.0.0.1', 9000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print("Сервер запущен...")

conn, addr = sock.accept()
print("Клиент подключился:", addr)

# Получаем контекст
ctx_size = int.from_bytes(recv_exact(conn, 4), 'big')
ctx_bytes = recv_exact(conn, ctx_size)
public_ctx = ts.context_from(ctx_bytes)

# Получаем зашифрованный (hidden + r)
vec_size = int.from_bytes(recv_exact(conn, 4), 'big')
enc_hidden = ts.ckks_vector_from(public_ctx, recv_exact(conn, vec_size))

# Вычисляем W2·(hidden + r) + b2
enc_logits = []
for i in range(10):
    score = enc_hidden.dot(W2[i].tolist()) + float(b2[i])
    enc_logits.append(score)

# Отправляем поштучно
for score in enc_logits:
    data = score.serialize()
    conn.sendall(len(data).to_bytes(4, 'big') + data)

conn.close()
print("Сервер завершил работу.")
