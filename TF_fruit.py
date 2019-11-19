from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, AveragePooling2D
from keras import backend, optimizers, losses, metrics
import numpy as np
from keras.preprocessing import image
from keras.callbacks import TensorBoard
import datetime
import tensorflow as tf

# tensorboard CLI entry: tensorboard --logdir=logs/ --host 192.168.1.73 --port 6006

# Setup
img_width, img_height = 256, 256  # the dimensions need to be declared first

training_data_folder = ""
validation_data_folder = ""

now = datetime.datetime.now()
time_now = f"{now.hour}.{now.minute}.{now.second}-{now.day}.{now.month}.{now.year}"
NAME = f"3-Fruit-CNN-{time_now}"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

number_of_training_samples = 35
number_of_validation_samples = 600
number_of_epochs = 30
the_batch_size = 5


def do_training():
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=8))  # Set to number of CPU cores

    # Generate the data
    if backend.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)  # 50/130,3

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    training_data_generator = train_datagen.flow_from_directory(
        training_data_folder,
        target_size=(img_width, img_height),
        batch_size=the_batch_size,
        class_mode="categorical",  # try sparse
    )

    validation_data_generator = train_datagen.flow_from_directory(
        validation_data_folder,
        target_size=(img_width, img_height),
        batch_size=the_batch_size,
        class_mode="categorical")

    # make the model itself
    the_model = Sequential([
        Dense(128, input_shape=input_shape, activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(5, 5), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=(5, 5), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(16, kernel_size=(5, 5), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(8, kernel_size=(5, 5), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(3, activation='softmax')
    ])

    the_model.summary()

    the_model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.adam(), metrics=["accuracy"])

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
    image_to_predict = image.load_img(location, target_size=(150, 150))
    image_to_predict = image.img_to_array(image_to_predict)
    image_to_predict = np.expand_dims(image_to_predict, axis=0)
    # Do data prediction
    the_result = the_model.predict(image_to_predict)

    objects = ["Apple", "Banana", "Orange"]

    performance = [round(the_result[0] * 100, 2),
                   round(the_result[1] * 100, 2),
                   round(the_result[2] * 100, 2)]

    output = [str(objects[0]) + " " + str(performance[0]) + "%",
              str(objects[1]) + " " + str(performance[1]) + "%",
              str(objects[2]) + " " + str(performance[2]) + "%"]


if __name__ == "__main__":
    do_training()
    predict_from_an_image()  # pass the model name "the_model" and the path to the test image
