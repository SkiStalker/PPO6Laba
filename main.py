import os
import typing

import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.datasets import load_files
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from keras.utils import np_utils


def load_dataset(data_path: str) -> tuple[typing.Any, typing.Any, typing.Any]:
    data_loading = load_files(data_path)
    files_add = np.array(data_loading['filenames'])
    load_targets = np.array(data_loading['target'])
    load_target_labels = np.array(data_loading['target_names'])
    return files_add, load_targets, load_target_labels


def convert_image_to_array_form(files: list[str]) -> list:
    images_array = []
    for file in files:
        images_array.append(img_to_array(load_img(file, color_mode='rgb',
                                                  target_size=(shape_of_img, shape_of_img))))
    return images_array


def tensorflow_based_model() -> Sequential:
    base_model = Sequential()
    base_model.add(Conv2D(filters=16, kernel_size=2,
                          input_shape=(100, 100, 3), padding='same'))
    base_model.add(Activation('relu'))
    base_model.add(MaxPooling2D(pool_size=2))
    base_model.add(Conv2D(filters=32, kernel_size=2,
                          activation='relu', padding='same'))
    base_model.add(MaxPooling2D(pool_size=2))
    base_model.add(Conv2D(filters=64, kernel_size=2,
                          activation='relu', padding='same'))
    base_model.add(MaxPooling2D(pool_size=2))
    base_model.add(Conv2D(filters=128, kernel_size=2,
                          activation='relu', padding='same'))
    base_model.add(MaxPooling2D(pool_size=2))
    base_model.add(Dropout(0.3))
    base_model.add(Flatten())
    base_model.add(Dense(150))
    base_model.add(Activation('relu'))
    base_model.add(Dropout(0.4))
    base_model.add(Dense(no_of_classes, activation='softmax'))
    return base_model


if __name__ == '__main__':
    pathToData = './data/plants/train/'
    pathToTest = './data/plants/test/'
    pathToValid = './data/plants/val/'
    shape_of_img = 100
    number_colors = 3
    x_train, y_train, target_labels = load_dataset(pathToData)

    x_test, y_test, _ = load_dataset(pathToTest)
    x_valid, y_valid, _ = load_dataset(pathToValid)
    no_of_classes = len(np.unique(y_train))
    y_train = np_utils.to_categorical(y_train, no_of_classes)
    y_test = np_utils.to_categorical(y_test, no_of_classes)
    y_valid = np_utils.to_categorical(y_valid, no_of_classes)
    x_train = np.array(convert_image_to_array_form(x_train))
    x_valid = np.array(convert_image_to_array_form(x_valid))
    x_test = np.array(convert_image_to_array_form(x_test))
    x_train = x_train.astype('float32') / 255
    x_valid = x_valid.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    model = tensorflow_based_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=20,
                        validation_data=(x_valid, y_valid),
                        verbose=2, shuffle=True)

    model.save('./plants_classifier.model')
    # model = models.load_model("./plants_classifier.model")
    loss_score, acc_score = model.evaluate(x_test, y_test)
    print('\n', 'Test loss:', loss_score)
    print('\n', 'Test accuracy:', acc_score)
    predictions = model.predict(x_test)
    fig = plt.figure(figsize=(8, 9))
    for i, idx in enumerate(np.random.choice(x_test.shape[0], size=8, replace=False)):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_test[idx]))
        pred_idx = np.argmax(predictions[idx])
        true_idx = np.argmax(y_test[idx])
        ax.set_title("{} ({})".format(target_labels[pred_idx],
                                      target_labels[true_idx]),
                     color=("green" if pred_idx == true_idx else "red"))

    plt.figure(1)

    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    directory = './data/raw/'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            img = np.array(convert_image_to_array_form([f]))
            img = img.astype('float32') / 255
            prediction = model.predict(img)
            index = np.argmax(prediction)
            print(f"Your image {os.path.basename(f)} is type { target_labels[index]}")
