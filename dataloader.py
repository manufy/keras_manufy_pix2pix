
from glob import glob
import scipy
import scipy.misc
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
import logging


class DataLoader():
    # def __init__(self, dataset_name, img_res=(128, 128)):
    def __init__(self, dataset_name, img_res=(128, 128)):
        logger.debug("DataLoader.constructor")
        print("scipy version: ", scipy.__version__)
        print("dataset_name: ", dataset_name)
        print("img_res_ ", img_res)
        self.dataset_name = dataset_name
        self.img_res = img_res

    def plot_image(self, legend, img):
        print(legend)
        imgplot = plt.imshow(img)
        plt.show()

    def load_images(self, batch_size=1, is_testing=False, random_select=True):
        # print("DataLoader.load_images")
        data_type = "train" if not is_testing else "test"
        path = glob('/Users/Manu/ML/Datasets/%s/%s/*' % (self.dataset_name, data_type))

        if random_select:
            batch_images = np.random.choice(path, size=batch_size)
        else:
            batch_images = ['/Users/Manu/ML/Datasets/facades/test\\87.jpg',
                            '/Users/Manu/ML/Datasets/facades/test\\30.jpg',
                            '/Users/Manu/ML/Datasets/facades/test\\15.jpg']

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = imageio.imread(img_path)
            # print("img imread shape: ", img.shape )
            # self.plot_image('Image imread',img)
            h, w, _ = img.shape
            _w = int(w / 2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            # print("img_A shape: " ,img_A.shape, " img_B shape: ", img_B.shape)

            # self.plot_image('Before resize img_A', img_A)
            # self.plot_image('Before resize img_B',img_B)

            # img_A = scipy.misc.imresize(img_A, self.img_res)
            # img_B = scipy.misc.imresize(img_B, self.img_res)

            # self.plot_image('After resize img_A',img_A)
            # self.plot_image('After resize img_B',img_B)

            # np.array(Image.fromarray(img_A).resize())

            # If training => do random flip

            if not is_testing and np.random.random() < 0.5:
                #   print("random flip")
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        # normalize Image [-1...1]

        imgs_A = np.array(imgs_A) / 127.5 - 1.
        imgs_B = np.array(imgs_B) / 127.5 - 1.

        # return array of images

        return imgs_A, imgs_B

    def load_images_batch(self, batch_size=1, is_testing=False):
        # print("DataLoader.load_images_batch")
        data_type = "train" if not is_testing else "val"
        path = glob('/Users/Manu/ML/Datasets/%s/%s/*' % (self.dataset_name, data_type))

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches - 1):
            batch = path[i * batch_size:(i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:

                # print ("reading ", img)

                # img = self.imread(img)

                img = imageio.imread(img)
                h, w, _ = img.shape
                half_w = int(w / 2)
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                # img_A = scipy.misc.imresize(img_A, self.img_res)
                # img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A) / 127.5 - 1.
            imgs_B = np.array(imgs_B) / 127.5 - 1.

            yield imgs_A, imgs_B

    # def imread(self, path):
    #   return scipy.misc.imread(path, mode='RGB').astype(np.float)

# dataloader python entry point standalone

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("keras_manu_dataloader_jupyter loaded")
#data_loader_test(