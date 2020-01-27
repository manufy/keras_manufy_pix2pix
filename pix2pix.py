
from __future__ import print_function, division
import os

from sqlalchemy.util import safe_reraise

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import scipy




from keras.datasets import mnist
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, model_from_json, load_model, model_from_yaml
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils import plot_model
import datetime
import matplotlib.pyplot as plt
from dataloader import DataLoader

import sys
import numpy as np

from glob import glob
import scipy.misc
from matplotlib.pyplot import imread
import numpy as np
from PIL import Image

import logging as logger

class Pix2Pix():
    def __init__(self):
        logger.debug("pix2pix constructor")
        self.frame = 0
        self.batch_size = 3
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.epochs = 200
        self.sample_interval = 50
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)
        # Number of filters in the first layer of G and D
        self.generator_filter_number = 64
        self.df = 64
        # Configure data loader
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)
        # Number of filters in the first layer of G and D
        self.generator_filter_number = 64
        self.df = 64

        self.optimizer = None
        self.discriminator = None
        self.generator = None
        self.combined = None

        self.discriminator_accuracy_graph = []
        self.generator_accuracy_graph = []
        self.discriminator_loss_graph = []
        self.generator_loss_graph = []

    def build_model(self):

        logger.debug("Building Neural Network Model")
        self.optimizer = Adam(0.0002, beta_1=0.5, beta_2=0.999)

        # build discriminator
        self.discriminator = self.build_discriminator()
        #print("Discriminator summary")
        #self.discriminator.summary()
        # Compile discriminator
        logger.debug("compiling discriminator")
        self.discriminator.compile(loss='mse',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        # Build generator
        self.generator = self.build_generator()
        #  print("Generator summary")
        #  self.discriminator.summary()
        # Input images and their conditioning images
        img_a = Input(shape=self.img_shape)
        img_b = Input(shape=self.img_shape)
        # By conditioning on B generate a fake version of A
        fake_a = self.generator(img_b)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_a, img_b])
        self.combined = Model(inputs=[img_a, img_b], outputs=[valid, fake_a])
        logger.debug("compiling combined")
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=self.optimizer,
                              metrics=['accuracy']
                              )
        ## print("Neural Network combined  model summary")
        #  self.combined.summary()
        plot_model(self.combined, to_file='model_combined.png')
        self.save()

    def serialize_model(self, model, filename):
        # serialize model
        json_string = model.to_json()
        with open(filename + ".json", "w") as json_file:
            json_file.write(json_string)

    def deserialize_model(self, filename):
        json_file = open(filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        return loaded_model

    def save(self):
        self.generator.save("generator.h5")
        self.discriminator.save("discriminator.h5")
        self.combined.save("combined.h5")

    def load(self):
        self.generator = load_model("generator.h5")
        self.discriminator = load_model("discriminator.h5")
        self.combined = load_model("combined.h5")

    def load_model(self):
        self.generator = self.deserialize_model('generator.json')
        self.discriminator = self.deserialize_model('discriminator.json')
        self.combined = self.deserialize_model('combined.json')

    def save_model(self):
        self.serialize_model(self.generator, 'generator')
        self.serialize_model(self.discriminator, 'discriminator')
        self.serialize_model(self.combined, 'combined')

    # Builds a Generator U-Net network type
    def build_generator(self):
        logger.debug("building Neural Network type U-Net Generator ...")
            
        def conv2d_downsample(layer_input, filters, f_size=4, batch_normalization=True):

            kernel_initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', kernel_initializer=kernel_initializer,
                       use_bias=False)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if batch_normalization:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d_upsample(layer_input, skip_input, filters, f_size=4, dropout_rate=0.0):

            kernel_initializer = RandomNormal(mean=0.0, stddev=0.02, seed=None)
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', kernel_initializer=kernel_initializer,
                       activation='relu')(u)
            if dropout_rate > 0.0:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)
        # print("input shape: ", d0.shape)

        # Downsampling
        d1 = conv2d_downsample(d0, self.generator_filter_number, batch_normalization=False)  # 64
        d2 = conv2d_downsample(d1, self.generator_filter_number * 2)  # 128
        d3 = conv2d_downsample(d2, self.generator_filter_number * 4)  # 256
        d4 = conv2d_downsample(d3, self.generator_filter_number * 8)  # 512
        d5 = conv2d_downsample(d4, self.generator_filter_number * 8)  # 512
        d6 = conv2d_downsample(d5, self.generator_filter_number * 8)  # 512
        d7 = conv2d_downsample(d6, self.generator_filter_number * 8)  # 512
        d8 = conv2d_downsample(d7, self.generator_filter_number * 8)  # 512

        # Upsampling

        u1 = deconv2d_upsample(d8, d7, self.generator_filter_number * 8, dropout_rate=0.5)  # 512
        u2 = deconv2d_upsample(u1, d6, self.generator_filter_number * 8, dropout_rate=0.5)  # 512
        u3 = deconv2d_upsample(u2, d5, self.generator_filter_number * 8, dropout_rate=0.5)  # 512
        u4 = deconv2d_upsample(u3, d4, self.generator_filter_number * 8)  # 512
        u5 = deconv2d_upsample(u4, d3, self.generator_filter_number * 4)  # 256
        u6 = deconv2d_upsample(u5, d2, self.generator_filter_number * 2)  # 128
        u7 = deconv2d_upsample(u6, d1, self.generator_filter_number)  # 64
        u8 = UpSampling2D(size=2)(u7)

        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u8)
        model = Model(d0, output_img)
        plot_model(model, to_file='model_generator.png')

        return model

    # Builds a Discriminator network type

    def build_discriminator(self):
        logger.debug("building Neural Network type Discriminator ...")

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_a = Input(shape=self.img_shape)
        img_b = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_a, img_b])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        model = Model([img_a, img_b], validity)
        plot_model(model, to_file='model_discriminator.png')
        return model

    # Train pix2pix network
    def train(self):
        start_time = datetime.datetime.now()

        # Adversarial loss output truths
        valid = np.ones((self.batch_size,) + self.disc_patch)
        fake = np.zeros((self.batch_size,) + self.disc_patch)

        for epoch in range(self.epochs):
            for batch_i, (imgs_a, imgs_b) in enumerate(self.data_loader.load_images_batch(self.batch_size)):

                # Condition on B and generate a translated version
                fake_a = self.generator.predict(imgs_b)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_a, imgs_b], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_a, imgs_b], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_a, imgs_b], [valid, imgs_a])
                print("g_loss: ", g_loss)
                elapsed_time = datetime.datetime.now() - start_time

                # get stats

                discriminator_loss = d_loss[0]
                discriminator_accuracy = 100 * d_loss[1]
                generator_loss = g_loss[0]
                generator_accuracy = [100 * g_loss[1]]

                # Plot the progress
                logger.debug("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s"
                             % (epoch, self.epochs, batch_i, self.data_loader.n_batches, discriminator_loss,
                                discriminator_accuracy,
                                generator_loss, elapsed_time))

                self.discriminator_accuracy_graph.append(discriminator_accuracy)
                self.generator_accuracy_graph.append(generator_accuracy)
                self.discriminator_loss_graph.append(discriminator_loss)
                self.generator_loss_graph.append(generator_loss)

                # If at save interval => save generated image samples
                if batch_i % self.sample_interval == 0:  # or self.frame == 0:
                    self.generate_frame()
                    self.generate_graph((epoch+1) * (batch_i+1))
                    self.save()

    def save_weights(self):
        self.serialize_weigths(self.generator,"generator_weights")
        self.serialize_weigths(self.discriminator, "discriminator_weights")
        self.serialize_weigths(self.combined, "combined_weightss")

    def load_weights(self):

        self.discriminator.load_weights("discriminator_weights.h5")
        self.generator.load_weights("generator_weights.h5")
        self.combined.load_weights("combined_weights.h5")

    def serialize_weigths(self, model, filename):
        model.save_weights(filename + ".h5")

    def deserialize_weigths(self, model, filename):
        model.load_weights(filename + ".h5")

    def generate_graph(self, epoch):

        plt.figure(0)
        # plt.plot(snn.history['acc'], 'r')
        # plt.plot(snn.history['val_acc'], 'g')

        plt.plot(self.discriminator_accuracy_graph, 'g')
        plt.plot(self.discriminator_loss_graph, 'r')
        # plt.plot(generator_accuracy_graph, 'y')
        plt.plot(self.generator_loss_graph, 'b')

        plt.xticks(np.arange(0, epoch, 2.0))
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.xlabel("Batchs/Epochs")
        plt.ylabel("Accuracy/Loss")
        plt.title("Accuracy vs Loss")
        plt.legend(['D_acc', 'D_loss', 'G_loss'])

        plt.savefig('evolution_graph.png')

    def generate_frame(self):

        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_a, imgs_b = self.data_loader.load_images(batch_size=3, is_testing=True, random_select=False)
        fake_a = self.generator.predict(imgs_b)

        # print("imgs_a",imgs_a)
        # print("fake_a", fake_a)

        gen_imgs = np.concatenate([imgs_b, fake_a, imgs_a])

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        gen_imgs = 1 * gen_imgs + 0.5

        titles = ['Entrada', 'Generado', 'Original']
        fig, axs = plt.subplots(r, c, figsize=(12, 12))
        cnt = 0
        for i in range(r):
            for j in range(c):
                #  print("imshow: gen_imgs: ", cnt)
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        # fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        self.frame += 1
        fig.savefig("images/%s/frame_%s.png" % (self.dataset_name, self.frame))
        plt.close()


# entry point

print("pix2pix neural network entry point")
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

neural_network = Pix2Pix()
###neural_network.build_model()
neural_network.load()

neural_network.train()


####

#neural_network.load_model()
#neural_network.load_weights()
