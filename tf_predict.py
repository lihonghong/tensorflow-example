#!/usr/bin/env python
# coding=gbk
# ==============================================================================
#          \file   binary_classification.py
#        \author   chenghuige  
#          \date   2015-11-30 16:06:52.693026
#   \Description  
# ==============================================================================

import tensorflow as tf
import numpy as np
import melt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('test', './corpus/feature.normed.rand.12000.1_2.txt', 'test file')
flags.DEFINE_string('method', 'logistic', 'currently support logistic/mlp')
# ----for mlp
flags.DEFINE_integer('hidden_size', 50, 'Hidden unit size')

trainset_file = FLAGS.train
testset_file = FLAGS.test

learning_rate = FLAGS.learning_rate
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size

testset = melt.load_dataset(testset_file)
print "finish loading test set ", testset_file
print 'num_features: ', testset.num_features
print 'testSet size: ', testset.num_instances()
print 'batch_size:', batch_size, ' learning_rate:', learning_rate, ' num_epochs:', num_epochs

trainer = melt.gen_binary_classification_trainer(testset)
py_x = Mlp().forward(trainer)
predict_op = tf.nn.sigmoid(py_x)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

teX, teY = testset.full_batch()

# Restore model weights from previously saved model
saver = tf.train.Saver()
model_path = "../tensorflow-model/sex_model"
load_path = saver.restore(sess, model_path)
print "Model restored from file: %s" % load_path
predicts = sess.run([predict_op], feed_dict=trainer.gen_feed_dict(teX, np.zeros(())))
print teY, predicts
# print 'final ', 'auc:', roc_auc_score(teY, predicts), 'cost:', cost_ / len(teY)
# print "Classification report for classifier %s\n" % (metrics.classification_report(teY, melt.p2c(predicts, 0.5)))
# print "Confusion matrix:\n%s" % metrics.confusion_matrix(teY, melt.p2c(predicts, 0.5))
