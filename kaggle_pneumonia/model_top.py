import argparse
import json

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop


def create_model(params, input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(params.drop_rate))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=RMSprop(lr=params.learning_rate),
                  loss='binary_crossentropy',
                  metrics=[keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score(), 'accuracy'])

    return model


def load_data(params):
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array([0] * 1341 + [1] * 3875)

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0] * 234 + [1] * 390)

    return (train_data, train_labels), (validation_data, validation_labels)


def train_model(model, train_data, train_labels, validation_data, validation_labels, params):
    return model.fit(train_data, train_labels,
                     epochs=params.epochs,
                     batch_size=params.batch_size,
                     validation_data=(validation_data, validation_labels),
                     callbacks=[TensorBoard(log_dir=params.log_dir)])


def main(params):
    (train_data, train_labels), (validation_data, validation_labels) = load_data(params)

    model = create_model(params, input_shape=train_data.shape[1:])

    if params.resume_run:
        model.load_weights(params.save_path)

    history = train_model(model,
                          train_data, train_labels,
                          validation_data, validation_labels,
                          params)

    json.dump(history.history, open(params.metrics_path, 'w'))
    model.save_weights(params.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nb_train_samples', type=int, default=5216,
                        help="Number of training samples.")
    parser.add_argument('--nb_val_samples', type=int, default=624,
                        help="Number of test samples.")
    parser.add_argument('--epochs', type=int, default=15,
                        help="Epochs of training.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size.")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="RMSprop learning rate.")
    parser.add_argument('--drop_rate', type=float, default=0.5,
                        help="Dense layer dropout rate.")
    parser.add_argument('--log_dir', type=str, default='logs/top_imagenet',
                        help="Where to save TensorBoard logs.")
    parser.add_argument('--metrics_path', type=str, default='fc_metrics.json',
                        help="Where to save json with metrics after training.")
    parser.add_argument('--save_path', type=str, default='fc_model.h5',
                        help="Where to save model weights after training.")
    parser.add_argument('--resume_run', action='store_const', const=True, default=False,
                        help='Load the model weights and continue training.')
    params = parser.parse_args()

    main(params)