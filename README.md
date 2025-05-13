# SecureAI
## Secure MNIST Digit Classification with Homomorphic Encryption

This project demonstrates secure digit classification using a neural network with homomorphic encryption. The client can draw a digit, which is then encrypted and sent to the server for private prediction.

## Features

- Secure Prediction: Uses TenSEAL for CKKS homomorphic encryption
- Interactive Drawing: GUI for digit input (holst.py)
- Split Neural Network:
    - Client computes first layer (784 → 128) locally
    - Server computes second layer (128 → 10) on encrypted data
- Adding random rave that makes it impossible to know the information

The secret key is located only on the client, so the result of the inference and the original image are known only to the client. The server performs the function of loading the model and processing the incoming encrypted request from the user.

## Installation

You need Python 3.13+ to run.

Install the dependencies and devDependencies and start the server.

```sh
git clone https://github.com/rinie1/SecureAI.git
cd SecureAI
pip install -r requirements.txt
python model_train.py
```

Your model files will be in 'mnist_model_split.npz' and 'model_params.json'.
Use .npz file for server and .json for client

```sh
python server.py -h
python client.py -h
```

There are some images to mnist in 'MNIST_JPG_testing'. But you can create your image, just don't use flag --image in client.

Example:
```sh
python server.py
python client.py
```

## Thanks
To [teavanist](https://github.com/teavanist/MNIST-JPG/commits?author=teavanist) for big database with MNIST images - [repo with images](https://github.com/teavanist/MNIST-JPG)

## Docker

---
