# A simple MNIST classifier (linear regression) with tensorflow
#

#
# Usage:
# python2.7 -m pip install tensorflow
# python3 board.py
# python3.5 board.py
#
# tensorboard --logdir=summary
# 
# JSCH 2017-04-23

from __future__ import print_function

import pickle

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.data import Dataset, Iterator
from tensorflow.python.platform import gfile




origin_path  = '/media/hhofmann/deeplearning/ilsvrc2012/LabelList_Heinz/'
    
def main():
    print("TensorFlow version ", tf.__version__)    
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        print("start save Weights as python lists ....\n")
        
#==============================================================================
#         Change here the paths, if you need new weights!!
#==============================================================================
        restorer=tf.train.import_meta_graph(origin_path + "../../getfingers_heinz/weights/7BnormBeforeRelu2.ckpt-00086000.meta")
        restorer.restore(sess,origin_path + "../../getfingers_heinz/weights/7BnormBeforeRelu2.ckpt-00103000")
        graph = tf.get_default_graph() 
        
        #saveStuff (getTensor ( Prepare the vector to get it by sess.run          )      (path, where the tensor shall be saved.)
        pickle.dump(sess.run(graph.get_tensor_by_name("1_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/1_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("3_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/3_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("5_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/5_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("6_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/6_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("7_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/7_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("8_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/8_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("10_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/10_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("11_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/11_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("12_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/12_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("13_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/13_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("14_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/14_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("15_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/15_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("16_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/16_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("17_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/17_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("18_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/18_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("19_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/19_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("21_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/21_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("22_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/22_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("23_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/23_conv_Layer_W_Variable.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("24_conv_Layer/W/Variable:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/24_conv_Layer_W_Variable.pkl","wb"))

        pickle.dump(sess.run(graph.get_tensor_by_name("1_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/1_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("3_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/3_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("5_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/5_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("6_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/6_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("7_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/7_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("8_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/8_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("10_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/10_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("11_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/11_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("12_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/12_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("13_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/13_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("14_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/14_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("15_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/15_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("16_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/16_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("17_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/17_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("18_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/18_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("19_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/19_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("21_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/21_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("22_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/22_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("23_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/23_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("24_conv_Layer/batch_norm/gamma/gamma:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/24_conv_Layer_batch_norm_gamma_gamma.pkl","wb"))
        
        
        pickle.dump(sess.run(graph.get_tensor_by_name("1_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/1_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("3_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/3_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("5_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/5_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("6_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/6_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("7_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/7_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("8_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/8_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("10_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/10_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("11_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/11_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("12_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/12_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("13_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/13_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("14_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/14_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("15_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/15_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("16_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/16_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("17_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/17_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("18_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/18_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("19_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/19_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("21_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/21_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("22_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/22_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("23_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/23_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        pickle.dump(sess.run(graph.get_tensor_by_name("24_conv_Layer/batch_norm/beta/beta:0")),open(origin_path + "../../getfingers_heinz/weights/pythonWeights/24_conv_Layer_batch_norm_beta_beta.pkl","wb"))
        print(sess.run(graph.get_tensor_by_name("1_conv_Layer/batch_norm/beta/beta:0")))
        print(pickle.load( open( origin_path + "../../getfingers_heinz/weights/pythonWeights/1_conv_Layer_batch_norm_beta_beta.pkl", "rb" ) ))
        
       

    print("finished")

    


if __name__ == "__main__":
   
    main()
