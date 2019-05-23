# https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia#chest_xray.zip

# data/ folder directory structure:
#     train/
#         NORMAL    - 1341 images
#         PNEUMONIA - 3875 images
#     val/
#         NORMAL    - 8 images
#         PNEUMONIA - 8 images
#     test/
#         NORMAL    - 234 images
#         PNEUMONIA - 390 images

import argparse
import logging
import warnings

from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , LearningRateScheduler
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score



def create_model(params):
    input_shape = (params.img_height, params.img_width, 3)

    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))  # calculate new density based on image size and quantity ?
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1))    # calculate new density based on image size and quantity ?
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=[keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score(), 'accuracy'])
    return model

def create_data_generators(params):
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=params.zoom_range,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        params.train_data_dir,
        target_size=(params.img_height, params.img_width),
        batch_size=params.batch_size,
        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
        params.test_data_dir,
        target_size=(params.img_height, params.img_width),
        batch_size=params.batch_size,
        class_mode='binary')

    return train_generator, test_generator


def train_model(model, train_generator, test_generator, params):
    # fit model
    return model.fit_generator(
        train_generator,
        steps_per_epoch=params.nb_train_samples // params.batch_size,
        epochs=params.epochs,
        validation_data=test_generator,
        validation_steps=params.nb_test_samples // params.batch_size,
        workers=params.workers,
        callbacks=[TensorBoard(log_dir=params.log_dir)])


def main(params):
    log = logging.getLogger('model-xrayslog')

    model = create_model(params)
    log.debug("Created model:")
    model.summary(print_fn=lambda x: log.debug(x))

    if params.resume_run:
        model.load_weights(params.save_path)
        log.info("Loaded weights from: %s", params.save_path)

    train_generator, test_generator = create_data_generators(params)
    log.info("Created training and test data generators.")

    train_model(model, train_generator, test_generator, params)
    log.info("Training finished!")

    model.save_weights(params.save_path)
    log.info("Saved weights to: %s", params.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_data_dir', type=str, default='data/train',
                        help="Path to directory with training data.")
    parser.add_argument('--nb_train_samples', type=int, default=5216,
                        help="Number of training samples.")
    parser.add_argument('--test_data_dir', type=str, default='data/test',
                        help="Path to directory with test data.")
    parser.add_argument('--nb_test_samples', type=int, default=624,
                        help="Number of test samples.")
    parser.add_argument('--img_height', type=int, default=150,
                        help="Images will be resized to this height.")
    parser.add_argument('--img_width', type=int, default=150,
                        help="Images will be resized to this width.")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Epochs of training.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size.")
    parser.add_argument('--workers', type=int, default=4,
                        help="Maximum number of that will execute the generator.")
    parser.add_argument('--zoom_range', type=float, default=0.1,
                        help="Range for random zoom: [1 - zoom_range, 1 + zoom_range].")
    parser.add_argument('--log_dir', type=str, default='logs/conv',
                        help="Where to save TensorBoard logs.")
    parser.add_argument('--save_path', type=str, default='model-xrayslog',
                        help="Where to save model weights after training.")
    parser.add_argument('--resume_run', action='store_const', const=True, default=False,
                        help='Load the model weights and continue training.')
    parser.add_argument('--debug', action='store_const', const=True, default=False,
                        help='Set debug logging level, otherwise info level is set.')
    params = parser.parse_args()

    # configure logger
    logger = logging.getLogger('model-xrayslog')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()  # console handler
    ch.setLevel(logging.DEBUG if params.debug else logging.INFO)
    ch.setFormatter(logging.Formatter(fmt='%(asctime)s [%(name)s:%(levelname)s]: %(message)s',
                                      datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # ignore warnings about bad EXIF
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    main(params)