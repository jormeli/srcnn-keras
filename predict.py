from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", help="The model to be used for prediction")
parser.add_argument("input", help="Input image file path")
parser.add_argument("output", help="Output image file path")
parser.add_argument("--baseline", help="Baseline bicubic interpolated image file path")
parser.add_argument("--scale", help="Scale factor", default=3.0, type=float)
args = parser.parse_args()

import numpy as np
from scipy import misc

from keras.models import load_model

input_size = 33
label_size = 21
pad = (33 - 21) / 2

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return rgb.dot(xform.T)

m = load_model(args.model)

X = misc.imread(args.input, mode='YCbCr')

w, h, c = X.shape
w -= int(w % args.scale)
h -= int(h % args.scale)
X = X[0:w, 0:h, :]
X[:,:,1] = X[:,:,0]
X[:,:,2] = X[:,:,0]

scaled = misc.imresize(X, 1.0/args.scale, 'bicubic')
scaled = misc.imresize(scaled, args.scale/1.0, 'bicubic')
newimg = np.zeros(scaled.shape)

if args.baseline:
    misc.imsave(args.baseline, scaled[pad : w - w % input_size, pad: h - h % input_size, :])

for i in range(0, h - input_size + 1, label_size):
    for j in range(0, w - input_size + 1, label_size):
        sub_img = scaled[j : j + input_size, i : i + input_size]

        prediction = m.predict(sub_img[None, None, :, :, 0]).reshape(label_size, label_size)
        newimg[j + pad : j + pad + label_size, i + pad : i + pad + label_size, 0] = prediction
        newimg[j + pad : j + pad + label_size, i + pad : i + pad + label_size, 1] = prediction
        newimg[j + pad : j + pad + label_size, i + pad : i + pad + label_size, 2] = prediction

newimg = newimg[pad : w - w % input_size, pad : h - h % input_size,:]
misc.imsave(args.output, newimg)
