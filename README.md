# VGG16 precision test

## vgg16_weights.npz

download from:
www.cs.toronto.edu/~frossard/tags/vgg16

## generate vgg16_fixed8bit/vggs.pb (float32)

python vgg16_fixed8bit/get_pbfile_from_npz.py

## generate quantized_graph.pb (fixed8)

* install **bazel**
* remember to cd into tensorflow folder when run 

	bazel build tensorflow/tools/quantization:quantize_graph
	bazel-bin/tensorflow/tools/quantization/quantize_graph --input=input_pbfile --output_node_names="softmax" --output=output_pbfile --mode=eightbit

* refer to: blog.csdn.net/xueyingxue001/article/details/72726421 for details

## precision

* vgg16_float16: top-5 85.726%, top-1 64.72%
* vgg16_float32: top-5 85.736%, top-1 64.722%
* vgg16_fixed8:  top-5

## experiment summary

* Transfering weights from float point to fixed point only make precision go down a little. 
* float16 and fixed8 run much slower than float32. float32 works 133 times faster than fixed8 on GPU. 


