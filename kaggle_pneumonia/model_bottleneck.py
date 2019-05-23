import argparse

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications


def main(params):
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        params.train_data_dir,
        target_size=(params.img_height, params.img_width),
        batch_size=params.batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, params.nb_train_samples // params.batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        params.val_data_dir,
        target_size=(params.img_height, params.img_width),
        batch_size=params.batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, params.nb_val_samples // params.batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_data_dir', type=str, default='data/train',
                        help="Path to directory with training data.")
    parser.add_argument('--nb_train_samples', type=int, default=5216,
                        help="Number of training samples.")
    parser.add_argument('--val_data_dir', type=str, default='data/test',
                        help="Path to directory with test data.")
    parser.add_argument('--nb_val_samples', type=int, default=624,
                        help="Number of test samples.")
    parser.add_argument('--img_height', type=int, default=150,
                        help="Images will be resized to this height.")
    parser.add_argument('--img_width', type=int, default=150,
                        help="Images will be resized to this width.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size.")
    params = parser.parse_args()

    main(params)
