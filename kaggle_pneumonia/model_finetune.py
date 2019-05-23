import argparse
import json
import keras_metrics

from keras import applications
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# Import functions for creating data generators and training model,
# those are the same and can be reused. Great!
from model_xrays import create_data_generators, train_model


def create_model(args):
    input_shape = (params.img_height, params.img_width, 3)

    # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(params.drop_rate))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(params.top_path)

    # add the model on top of the convolutional base
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[1:15]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=params.learning_rate, momentum=0.9),
                  metrics=[keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score(), 'accuracy'])

    return model


def main(params):
    model = create_model(params)

    train_generator, validation_generator = create_data_generators(params)

    history = train_model(model, train_generator, validation_generator, params)

    json.dump(history.history, open(params.metrics_path, 'w'))
    model.save_weights(params.save_path)


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
    parser.add_argument('--epochs', type=int, default=15,
                        help="Epochs of training.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size.")
    parser.add_argument('--workers', type=int, default=4,
                        help="Maximum number of that will execute the generator.")
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help="Momentum(0.9) learning rate.")
    parser.add_argument('--drop_rate', type=float, default=0.5,
                        help="Dense layer dropout rate.")
    parser.add_argument('--shear_range', type=float, default=0.2,
                        help="Shear intensity (angle) in counter-clockwise direction in degrees.")
    parser.add_argument('--zoom_range', type=float, default=0.2,
                        help="Range for random zoom: [1 - zoom_range, 1 + zoom_range].")
    parser.add_argument('--log_dir', type=str, default='logs/fine_tune',
                        help="Where to save TensorBoard logs.")
    parser.add_argument('--metrics_path', type=str, default='vgg_metrics.json',
                        help="Where to save json with metrics after training.")
    parser.add_argument('--top_path', type=str, default='fc_model.h5',
                        help="Where to load the top model weights from.")
    parser.add_argument('--save_path', type=str, default='vgg_model.h5',
                        help="Where to save model weights after training.")
    params = parser.parse_args()

    main(params)
