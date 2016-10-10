#!/usr/bin/env python
# coding=gbk
# ==============================================================================
# python ./tf_predict.py  --modelPath /tmp/model/sex.model --test /tmp/sex.test --predictPath /tmp/sex.predict —-numClass 1
# ==============================================================================

import tensorflow as tf
import numpy as np
import melt
import model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('modelPath', '', 'path of saving train model')
flags.DEFINE_string('test', '', 'test file')
flags.DEFINE_string("predictPath", '', "path of saving predict data")

flags.DEFINE_integer('hidden_size', 50, 'Hidden unit size')
flags.DEFINE_integer('num_class', 1, 'output unit size')
flags.DEFINE_string("optimizer", "sgd", "optimizer to train: sgd,momentum,adadelta,adagrad,adam,ftrl,rmsprop")
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('gpu', 0, 'use which gpu')

modelPath = FLAGS.modelPath
predictPath = FLAGS.predictPath
testset_file = FLAGS.test
numClass = FLAGS.num_class
gpu = FLAGS.gpu

testset = melt.load_dataset(testset_file)
print "finish loading test set ", testset_file
print 'num_features: ', testset.num_features
print 'testSet size: ', testset.num_instances()
print 'gpu: ', gpu
print 'model path: ', modelPath
print 'predict path: ', predictPath
print 'num class: ', numClass

tester = melt.gen_multi_classification_trainer(testset, numClass)
py_x = model.Mlp().forward(tester, FLAGS, numClass, gpu)
predict_op = tf.nn.sigmoid(py_x)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
init = tf.initialize_all_variables()
sess.run(init)

teX, teY = testset.full_batch()

# Restore model weights from previously saved model
saver = tf.train.Saver()
load_path = saver.restore(sess, modelPath)
print "Model restored from file: %s" % load_path
predicts = sess.run([predict_op], feed_dict=tester.gen_feed_dict(teX, np.zeros((testset.num_instances(), numClass))))

if predictPath != '':
    np.savetxt(predictPath, np.c_[teY, predicts[0]], delimiter=',', comments='',
               fmt='%s%s' % ('%s', ',%s' * numClass))
    print("Predict result saved in :%s", predictPath)
