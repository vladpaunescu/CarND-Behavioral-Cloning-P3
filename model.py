import argparse
import csv
from argparse import _ActionsContainer
from ast import parse

import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import random

import sklearn


DATASET_DIR = "../training_t1/"
IMG_DIR = DATASET_DIR + "IMG/"

# center left right
CORRECTIONS = [0.0, +0.2, -0.2]

SHIFT_X_MAX = 60
SHIFT_Y_MAX = 20

def get_samples(skip_first=False):
    lines = []
    with open(DATASET_DIR + "driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    if skip_first:
        return lines[1:]

    return lines


def augment_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = v.astype(np.float64)
    v *= np.random.uniform(low=0.5, high=1.5)
    v[v > 255] = 255
    v = v.astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def augment_flip(img, angle):
    flip_choice = np.random.randint(2)
    if flip_choice == 0:
        return img, angle

    return cv2.flip(img, 1), -angle


def augment_shift(img, angle):
    random_hshift = np.random.uniform(low=-SHIFT_X_MAX, high=SHIFT_X_MAX)
    random_vshift = np.random.uniform(low=-SHIFT_Y_MAX, high=SHIFT_Y_MAX)
    M = np.float32([[1, 0, random_hshift], [0, 1, random_vshift]])
    rows, cols, depth = img.shape
    image = cv2.warpAffine(img, M, (cols, rows))
    angle += random_hshift * 0.3 / SHIFT_X_MAX

    return image, angle


def train_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []


            for batch_sample in batch_samples:
                left_right_center_choice = np.random.randint(3)
                img_path = batch_sample[left_right_center_choice]
                name = IMG_DIR + img_path.split('/')[-1]
                image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                angle = center_angle + CORRECTIONS[left_right_center_choice]

                image = augment_brightness(image)

                image, angle = augment_flip(image, angle)
                image, angle = augment_shift(image, angle)

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def valid_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []


            for batch_sample in batch_samples:
                left_right_center_choice = np.random.randint(3)
                img_path = batch_sample[left_right_center_choice]
                name = IMG_DIR + img_path.split('/')[-1]
                image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                angle = center_angle + CORRECTIONS[left_right_center_choice]

                images.append(image)
                angles.append(angle)

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


def get_cropped_sample(img):
    crop = img[70:-25, :]
    cv2.imwrite("crop_sample.png", crop)
    return crop


def get_left_right_center_samples(sample):
    names = ["center.png", "left.png", "right.png"]
    for i in range(3):
        path = sample[i]
        img = cv2.imread(IMG_DIR + path.split('/')[-1])
        cv2.imwrite(names[i], img)

def get_flip_sample(img):
    flipped = cv2.flip(img, 1)
    cv2.imwrite("original.png", img)
    cv2.imwrite("flipped.png", flipped)
    return flipped

def get_brightness_augmentation(img):
    bright = augment_brightness(img)
    cv2.imwrite("brightness_augmentation.png", bright)
    return bright

def get_shift_augmentation(img):
    shifted_img, angle  = augment_shift(img, 0)
    cv2.imwrite("shift_augmentation.png", shifted_img)
    return shifted_img


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--log_augmentations', dest='log_augmentations', help='If should log augmentation samples',
                        default=False, type=bool)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # samples = get_samples(skip_first=True)
    samples = get_samples(skip_first=False)
    print(len(samples))

    args = parse_args()
    if args.log_augmentations:
        print("Logging augmentation samples")
        sample = samples[10][0]
        name = IMG_DIR + sample.split('/')[-1]
        image = cv2.imread(name)
        crop = img_path = get_cropped_sample(image)
        get_left_right_center_samples(samples[10])
        get_flip_sample(image)
        get_brightness_augmentation(image)
        get_shift_augmentation(image)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = train_generator(train_samples, batch_size=256)
    validation_generator = valid_generator(validation_samples, batch_size=32)

    model = NvidiaModel()
    model.compile(loss="mse", optimizer="adam")
    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
    model.fit_generator(train_generator, samples_per_epoch=20000, validation_data=validation_generator,
                        nb_val_samples=len(validation_samples), nb_epoch=7)

    model.save('model_nvidia_generator_shift_t3.h5')



