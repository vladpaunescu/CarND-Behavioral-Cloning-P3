import csv
from argparse import _ActionsContainer

import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import random

import sklearn


DATASET_DIR = "../data/"
IMG_DIR = DATASET_DIR + "IMG/"

# center left right
CORRECTIONS = [0.0, +0.2, -0.2]
def get_samples():
    lines = []
    images = []
    measurements = []
    with open(DATASET_DIR + "driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)


    # skip table header
    return lines[1:]


# for line in lines[1:]:
#     measurement = float(line[3])
#     for i in range(3):
#         rel_measurement = measurement + CORRECTIONS[i]
#
#         img_path = line[i]
#         fn = img_path.split('/')[-1]
#         image = cv2.imread(IMG_DIR + fn)
#
#         images.append(image)
#         measurements.append(rel_measurement)
#
#         images.append(cv2.flip(image, 1))
#         measurements.append(-rel_measurement)
#
# X_train = np.array(images)
# y_train = np.array(measurements)

def generator(samples, batch_size=32):

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            left_right_center_choice = np.random.randint(0, 3, size=batch_size)
            flip_choice = np.random.randint(0, 2, size=batch_size)
            # shift_choice = np.random.randint(0, 4, size=batch_size)

            for i, batch_sample in enumerate(batch_samples):
                img_path = batch_sample[left_right_center_choice[i]]
                name = IMG_DIR + img_path.split('/')[-1]
                image = cv2.imread(name)
                rows, cols, depth = image.shape
                center_angle = float(batch_sample[3])
                angle = center_angle + CORRECTIONS[left_right_center_choice[i]]

                hshift_amount = random.random() * 2 - 1
                vshift_amount = random.random() * 2 - 1
                M = np.float32([[1, 0, hshift_amount * 100], [0, 1, vshift_amount * 20]])
                image = cv2.warpAffine(image, M, (cols, rows))
                angle = angle + hshift_amount / 4.0
                # M = None
                # if shift_choice[i] == 1:
                #     # left shift
                #     M = np.float32([[1, 0, -shift_amount * 100], [0, 1, 0]])
                #     angle = angle - shift_amount / 4.0
                # elif shift_choice[i] == 2:
                #     # right shift
                #     M = np.float32([[1, 0, shift_amount * 100], [0, 1, 0]])
                #     angle = angle + shift_amount / 4.0
                # elif shift_choice[i] == 3:
                #     # down shift
                #     M = np.float32([[1, 0, 0], [0, 1, shift_amount * 20]])
                # elif shift_choice[i] == 4:
                #     # up shift
                #     M = np.float32([[1, 0, 0], [0, 1, -shift_amount * 20]])
                #
                # if M is not None:
                #     image = cv2.warpAffine(image, M, (cols, rows))

                if flip_choice[i] == 0:
                    images.append(image)
                    angles.append(angle)
                else:
                    images.append(cv2.flip(image, 1))
                    angles.append(-angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def LeNetModel():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 3), output_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def NvidiaModel():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 3), output_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    return model


if __name__ == "__main__":
    samples = get_samples()
    print(len(samples))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=256)
    validation_generator = generator(validation_samples, batch_size=32)

    model = NvidiaModel()
    model.compile(loss="mse", optimizer="adam")
    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
    model.fit_generator(train_generator, samples_per_epoch=40000, validation_data = validation_generator,
                        nb_val_samples = len(validation_samples), nb_epoch=10)

    model.save('model_nvidia_generator_shift.h5')



