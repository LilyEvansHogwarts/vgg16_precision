# VGG16 (8-bit fixed point)



## test_pbfile_generator

test1.py test2.py test3.py are three ways to generate a **vggs.pb** file, which contains vgg16 test graph.

It is worth notice that this file only generate a test graph from vgg16_weights.npz without extra training.

Basically, this file generate a .pb file from .npz file without any change to the original data.

## method

* generate vggs.pb from test_pb_generator
* generate a 8-bit fixed point test graph vgg16_test_graph.pb from vggs.pb with **bazel** command
* get the baseline test precision with test_vgg.py
