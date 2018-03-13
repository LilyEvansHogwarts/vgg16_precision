# VGG16 precision test

## vgg16_weights.npz

download from:
www.cs.toronto.edu/~frossard/tags/vgg16

## generate vggs.pb (float32)

python get_pbfile_from_npz.py

## generate quantized_graph.pb (fixed8)

* remember to first download tensorflow
   git clone https://github.com/tensorflow/tensorflow.git
* cd tensorflow
* ./configure
* bazel build --config=mkl --copt=
