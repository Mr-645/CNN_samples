from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend
import numpy as np
from keras.preprocessing import image
from keras.callbacks import TensorBoard
from keras.datasets import imdb  # data loading --- train_data=imdb.load_data('yourdatapath')
import time

# tensorboard CLI entry: tensorboard --logdir=logs/ --host 192.168.1.73 --port 6006

# Setup
img_width, img_height = 150, 150

training_data_folder = "C:/Users/drift/Downloads/data/train"
validation_data_folder = "C:/Users/drift/Downloads/data/validation"

# training_data_folder = "C:/Users/drift/Downloads/ImageAssistant Batch Image Downloader"
# validation_data_folder = "C:/Users/drift/Downloads/ImageAssistant Batch Image Downloader"

# NAME = "Cats-vs-dogs-CNN-{}".format(int(time.time()))
NAME = "5-Fruit-CNN-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

number_of_training_samples = 130  # 1000
number_of_validation_samples = 130  # 100
number_of_epochs = 30  # 50, try 31
the_batch_size = 130  # 20


def do_training():
    # Generate the data
    if backend.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)  # 50/130,3

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)  # for testing only

    training_data_generator = train_datagen.flow_from_directory(
        training_data_folder,
        target_size=(img_width, img_height),
        batch_size=the_batch_size,
        class_mode="binary")  # was binary before

    validation_data_generator = train_datagen.flow_from_directory(
        validation_data_folder,
        target_size=(img_width, img_height),
        batch_size=the_batch_size,
        class_mode="binary")  # was binary before

    # make the model itself
    the_model = Sequential()
    the_model.add(Convolution2D(32, (3, 3), activation="relu", input_shape=input_shape))  # number of features and 3x3 pixel matrix
    the_model.add(MaxPooling2D(pool_size=(2, 2)))

    the_model.summary()

    the_model.add(Convolution2D(32, (3, 3), activation="relu"))
    the_model.add(MaxPooling2D(pool_size=(2, 2)))

    the_model.add(Convolution2D(64, (3, 3), activation="relu"))
    the_model.add(MaxPooling2D(pool_size=(2, 2)))

    the_model.add(Flatten())  # 2D to 1D image
    the_model.add(Dense(units=64, activation="relu"))
    the_model.add(Dropout(0.5))
    the_model.add(Dense(1, activation="sigmoid"))

    the_model.summary()
    the_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    # the_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # the training itself
    the_model.fit_generator(  # connects nodes
        training_data_generator,
        steps_per_epoch=number_of_training_samples // the_batch_size,
        epochs=number_of_epochs,
        validation_data=validation_data_generator,
        validation_steps=number_of_validation_samples // the_batch_size,
        callbacks=[tensorboard],
        verbose=1,
    )
    return the_model


def predict_from_an_image(the_model, location):
    # print("\r" + str(location))
    # Feed it an image to do a prediction on
    image_to_predict = image.load_img(location, target_size=(150, 150))
    image_to_predict = image.img_to_array(image_to_predict)
    image_to_predict = np.expand_dims(image_to_predict, axis=0)
    # Do data prediction
    the_result = the_model.predict(image_to_predict)
    return the_result
    
the_result = m.predict_from_an_image(
    the_model=my_model_1,
    location="C:/Users/drift/Desktop/dog.jpg")

if the_result[0][0] == 1:
    the_prediction = "dog"
else:
    the_prediction = "cat"

print(the_result)
print(the_prediction)
