# SecureAI
## Cryptographic protocol for secure NN inference

---

SecureAI is cryptographic protocol for secure neural network inference
We using MNIST model here. You can try another model but this is beyond the scope of this project

## Features

- Import any 28x28 picture with digit.

The secret key is located only on the client, so the result of the inference and the original image are known only to the client.
The server performs the function of loading the model and processing the incoming encrypted request from the user.

## Installation

You need Python 3.13+ to run this.

Install the dependencies and start the 'model_train.py'
Your model would be in 'mnist_model_split.npz'.
Then you need to start 'server.py' and 'clinet.py'.
There are some images to mnist in 'MNIST_JPG_testing'.

```sh
git clone https://github.com/rinie1/SecureAI.git
cd SecureAI
pip install -r requirements.txt
python model_train.py
python server.py
python client.py
```

## Thanks
To [teavanist](https://github.com/teavanist/MNIST-JPG/commits?author=teavanist) for big database with MNIST images - [repo with images](https://github.com/teavanist/MNIST-JPG)

## Docker

---