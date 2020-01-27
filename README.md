# keras_manufy_pix2pix

Deep Learning pix2pix

- Learning video demo at https://youtu.be/04tMpEpjKQs
- What can pix2pix do ? at https://phillipi.github.io/pix2pix/images/teaser_v3.png
- Added dropout
- Changed hyperparameters for encoder/decoder

REQUISITES:

AMD RADEON GPU to go parallel on training phase and gain x10 performance over CPU

DATASET:

Must be in a folder /images/facades from facades dataset

USE with PlaidML to support AMD RADEON GPU:

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
neural_network = Pix2Pix()
neural_network.build_model()
neural_network.train()

from command line:

python pix2pix.py

*TODO FOR LOAD MODELS*

Save / Load training process. Gives an error loading .h5 deep network definition and weights.

UserWarning: No training configuration found in save file: the model was *not* compiled. Compile it manually.
warnings.warn('No training configuration found in save file: '
UserWarning: Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
warnings.warn('Error in loading the saved optimizer '
