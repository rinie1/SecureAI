# SecureAI
## Secure MNIST Digit Classification with Homomorphic Encryption

Cryptographic protocol for secure neural network inference.

## Features

- Secure Prediction: Uses TenSEAL for CKKS homomorphic encryption
- Split Neural Network:
    - Client computes first layer (784 → 128) locally
    - Server computes second layer (128 → 10) on encrypted data
- Adding random rave that makes it impossible to know the information
- Interactive Drawing: GUI for digit input (holst.py)
- Docker for server

The decryption keys are located only on the client, the server and other third parties cannot view the contents of the inference. The cryptographic protocol combines several encryption methods, which makes it a reliable tool.

## Installation

You need Python 3.10+ to run.

Install the dependencies (requirements.txt) and train your model or use ready.

```sh
git clone https://github.com/rinie1/SecureAI.git
cd SecureAI
pip install -r requirements.txt
python model_train.py
```

Your model files will be in 'mnist_model_split.npz' and 'model_params.json'.
Use .npz file for server and .json for client. Files must be in the appropriate directories.

Help information:
```sh
python server.py -h
python client.py -h
```

There are some images in 'MNIST_JPG_testing'. But you can create your image, just don't use flag --image when launching the 'client.py'.

Example (for Linux only):
1. Server (local requirements):
```sh
cd SecureAI/server/
pip install -r requirements.txt
python server.py
```
2. Client (local requirements, another terminal):
```sh
cd SecureAI/client/
pip install -r requirements.txt
python client.py
```
\
Example (for Windows):
1. Server (local requirements):
```sh
cd SecureAI/server/
pip install -r requirements.txt
python server.py --host 127.0.0.1
```
2. Client (local requirements, another terminal):
```sh
cd SecureAI/client/
pip install -r requirements.txt
python client.py --host 127.0.0.1
```

Also you can choose your own port - just use --port [port] (-p) (default 5000).
If you are using your own model's weights, use --model (-m) [name.npz] flag on server and --model (-m) [name.json] on client.
Server saves logs in 'logs.txt', but you can save them in another file, use --file (-f) [file_name.txt].

## Docker
Use Docker for building server image.\
Docker version 28.1.1
```sh
cd SecureAI/
docker build -t secureai-server .
```
Run Docker Image:
```sh
docker run -it -p 5000:5000 --name secureai-server secureai-server
```
Server logs
```sh
docker logs secureai-server
```
Stop secureai-server and remove it
```sh
docker stop secureai-server
docker rm secureai-server
```

## Thanks
To [teavanist](https://github.com/teavanist/MNIST-JPG/commits?author=teavanist) for big database with MNIST images - [repo with images](https://github.com/teavanist/MNIST-JPG)
