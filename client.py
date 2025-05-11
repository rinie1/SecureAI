import socket
import numpy as np
from PIL import Image
import tenseal as ts

def recv_exact(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise ConnectionError("Connection closed")
        data += packet
    return data

# Загрузка fc1 и параметров для маски
data = np.load("mnist_model_split.npz")
W1 = data['W1']   # (128, 784)
b1 = data['b1']   # (128,)
W2 = data['W2']   # (10, 128) — нужно на клиенте для снятия маски
b2 = data['b2']   # (10,)

# Обработка изображения
img = Image.open("MNIST_JPG_testing/3/3300.jpg").convert('L').resize((28, 28))
arr = np.array(img, dtype=np.float32) / 255.0
arr = (arr - 0.5) / 0.5
input_vec = arr.flatten()  # (784,)

# Локальный fc1 + ReLU
hidden = np.dot(W1, input_vec) + b1
hidden = np.maximum(hidden, 0.0)

r = np.random.randn(hidden.shape[0])
# Считаем заранее на клиенте W2·r, чтобы позже снять маску
Wr = W2.dot(r)  # (10,)
# Контекст CKKS
context = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[40, 20, 40])
context.global_scale = 2**20
context.generate_galois_keys()
# Зашифровываем hidden + маску r
enc_hidden = ts.ckks_vector(context, (hidden + r).tolist())
# Отправляем контекст и зашифрованный вектор
HOST, PORT = '127.0.0.1', 9000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

ctx_bytes = context.serialize(save_secret_key=False)
sock.sendall(len(ctx_bytes).to_bytes(4, 'big'))
sock.sendall(ctx_bytes)

enc_bytes = enc_hidden.serialize()
sock.sendall(len(enc_bytes).to_bytes(4, 'big'))
sock.sendall(enc_bytes)

# Получаем зашифрованные логиты
enc_logits = []
for _ in range(10):
    length = int.from_bytes(recv_exact(sock, 4), 'big')
    data = recv_exact(sock, length)
    enc_score = ts.ckks_vector_from(context, data)
    enc_logits.append(enc_score)
sock.close()

# Дешифруем, снимаем маску и предсказываем
logits = np.array([score.decrypt()[0] for score in enc_logits])
# Логиты пришли замаскированные: logits = W2·(hidden + r) + b2
# Вычитаем W2·r:
logits_unmasked = logits - Wr
pred = int(np.argmax(logits_unmasked))
print("Предсказанный класс:", pred)
