import tensorflow as tf
import os
import  numpy as np
import PIL.Image as Image
from skimage import io, transform
from scipy.misc import imread, imresize


def recognize(jpg_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input:0")
#            print input_x
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
#           print out_softmax
#            out_label = sess.graph.get_tensor_by_name("output:0")
#           print "out_label",out_label

            f = os.listdir(jpg_path)
            for eachfile in f:
                img = imread(jpg_path+eachfile,mode='RGB')
                img = imresize(img, (224,224))
                img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(img,[-1,224,224,3])})
                preds = np.argsort(img_out_softmax)[0][995:1000]
                print eachfile, preds


'''
            img = imread(jpg_path, mode='RGB')
            img = imresize(img, (224, 224))
            img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(img, [-1, 224, 224, 3])})

            print "img_out_softmax:",img_out_softmax
            preds = np.argsort(img_out_softmax)[0][995:1000]
            print "preds:", preds
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print "label:",prediction_labels
'''

recognize("img/t1/", "quantized_graph.pb")
recognize("img/t2/", "quantized_graph.pb")
recognize("img/t3/", "quantized_graph.pb")
recognize("img/t4/", "quantized_graph.pb")
recognize("img/t5/", "quantized_graph.pb")
recognize("img/t6/", "quantized_graph.pb")
recognize("img/t7/", "quantized_graph.pb")
recognize("img/t8/", "quantized_graph.pb")
recognize("img/t9/", "quantized_graph.pb")
recognize("img/t10/", "quantized_graph.pb")
recognize("img/t11/", "quantized_graph.pb")
recognize("img/t12/", "quantized_graph.pb")
recognize("img/t13/", "quantized_graph.pb")
recognize("img/t14/", "quantized_graph.pb")
recognize("img/t15/", "quantized_graph.pb")
recognize("img/t16/", "quantized_graph.pb")
recognize("img/t17/", "quantized_graph.pb")
recognize("img/t18/", "quantized_graph.pb")
recognize("img/t19/", "quantized_graph.pb")
recognize("img/t20/", "quantized_graph.pb")
