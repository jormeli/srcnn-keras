import sys
from os import listdir
from os.path import isfile, join, exists

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save", help="Path to save the checkpoints to")
parser.add_argument("data", help="Training data directory")
args = parser.parse_args()

input_dir = join(args.data, "input")
label_dir = join(args.data, "label")

if not (exists(input_dir) and exists(label_dir)):
    print("Input/label directories not found")
    sys.exit(1)

import numpy as np
from scipy import misc

from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

inputs = Input(shape=(1, 33, 33))
x = Convolution2D(64, 9, 9, input_shape=(1, 33, 33), activation='relu', init='he_normal')(inputs)
x = Convolution2D(32, 1, 1, activation='relu', init='he_normal')(x)
x = Convolution2D(1, 5, 5, init='he_normal')(x)
m = Model(input=inputs, output=x)
m.compile(Adam(lr=0.001), 'mse')

X = np.array([misc.imread(join(input_dir, f))[None,:,:,0] for f in listdir(input_dir)])
y = np.array([misc.imread(join(label_dir, f))[None,:,:,0] for f in listdir(label_dir)])

count = 1
while True:
    m.fit(X, y, batch_size=128, nb_epoch=5)
    if args.save:
        print("Saving model " + str(count * 5))
        m.save(join(args.save, 'model_' + str(count * 5) + '.h5'))
    count += 1
