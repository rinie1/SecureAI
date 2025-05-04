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

# Загрузка fc1
data = np.load("mnist_model_split.npz")
W1 = data['W1']  # (128, 784)
b1 = data['b1']  # (128,)

# Загрузка изображения
img = Image.open("MNIST_JPG_testing/0/3.jpg").convert('L').resize((28, 28))
arr = np.array(img, dtype=np.float32) / 255.0
arr = (arr - 0.5) / 0.5  # та же нормализация, что и при обучении
input_vec = arr.flatten()  # (784,)

# Вычисляем fc1 (вручную), применяем ReLU
hidden = np.dot(W1, input_vec) + b1
hidden = np.maximum(hidden, 0.0)  # ReLU
hidden_list = hidden.tolist()

# Гомоморфное шифрование
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[40, 20, 40])
context.global_scale = 2**20
context.generate_galois_keys()
enc_hidden = ts.ckks_vector(context, hidden_list)

# Отправляем на сервер
HOST = '127.0.0.1'
PORT = 9000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

ctx_bytes = context.serialize(save_secret_key=False)
sock.sendall(len(ctx_bytes).to_bytes(4, byteorder='big'))
sock.sendall(ctx_bytes)

enc_bytes = enc_hidden.serialize()
sock.sendall(len(enc_bytes).to_bytes(4, byteorder='big'))
sock.sendall(enc_bytes)

# Получаем логиты
enc_logits = []
for _ in range(10):
    length = int.from_bytes(recv_exact(sock, 4), byteorder='big')
    data = recv_exact(sock, length)
    enc_score = ts.ckks_vector_from(context, data)
    enc_logits.append(enc_score)

sock.close()

# Расшифровываем и предсказываем
logits = [score.decrypt()[0] for score in enc_logits]
print("Предсказанный класс:", int(np.argmax(logits)))
