# keras_manufy_pix2pix

Deep Learning pix2pix

- Added dropout
- Changed hyperparameters for encoder/decoder

REQUISITES:

AMD RADEON GPU to go parallel on training phase and gain x10 performance over CPU

DATASET:

Must be in a folder /images/facades from facades dataset

USE:


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

 # Use PlaidML to support AMD RADEON GPU
neural_network = Pix2Pix()

neural_network.build_model()
n


neural_network.train()



python pix2pix.py

TODO FOR LOAD MODELS:

Save / Load training process. Gives an error loading .h5 deep network definition and weights.

UserWarning: No training configuration found in save file: the model was *not* compiled. Compile it manually.
warnings.warn('No training configuration found in save file: '
UserWarning: Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
warnings.warn('Error in loading the saved optimizer '
