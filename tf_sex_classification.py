#!/usr/bin/env python
# coding=gbk
# ==============================================================================
# run :
# python ./tf_sex_classification.py --train sex.train --test sex.validation
# ==============================================================================
import sys

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import melt
import model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 60, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 500, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train', '', 'train file')
flags.DEFINE_string('test', '', 'test file')
flags.DEFINE_string('method', 'mlp', 'currently support mlp')
flags.DEFINE_integer('hidden_size', 50, 'Hidden unit size')
flags.DEFINE_integer('num_class', 1, 'output unit size')
flags.DEFINE_integer('gpu', 0, 'use which gpu')
flags.DEFINE_string('modelPath', '', 'path of saving train model')
flags.DEFINE_string("optimizer", "sgd", "optimizer to train: sgd,momentum,adadelta,adagrad,adam,ftrl,rmsprop")

trainset_file = FLAGS.train
testset_file = FLAGS.test

learning_rate = FLAGS.learning_rate
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
numClass = FLAGS.num_class
method = FLAGS.method
gpu = FLAGS.gpu
modelPath = FLAGS.modelPath

print 'gpu: ', gpu
print 'model path: ', modelPath

trainset = melt.load_dataset(trainset_file)
print "finish loading train set ", trainset_file
testset = melt.load_dataset(testset_file)
print "finish loading test set ", testset_file

assert (trainset.num_features == testset.num_features)
num_features = trainset.num_features
print 'num_features: ', num_features
print 'trainSet size: ', trainset.num_instances()
print 'testSet size: ', testset.num_instances()
print 'batch_size:', batch_size, ' learning_rate:', learning_rate, ' num_epochs:', num_epochs

trainer = melt.gen_binary_classification_trainer(trainset)

py_x = model.Mlp().forward(trainer, FLAGS, numClass, gpu)
Y = trainer.Y

cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(py_x, Y))
train_op = model.getOptimizer(FLAGS, learning_rate).minimize(cost)  # construct optimizer
predict_op = tf.nn.sigmoid(py_x)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
init = tf.initialize_all_variables()
sess.run(init)

teX, teY = testset.full_batch()

num_train_instances = trainset.num_instances()
thread = 0.5
for i in range(num_epochs):
    predicts, cost_ = sess.run([predict_op, cost], feed_dict=trainer.gen_feed_dict(teX, teY))
    print i, 'auc:', roc_auc_score(teY, predicts), 'cost:', cost_ / len(teY)
    print 'logloss:', melt.logloss(teY, predicts)
    print "Classification report for classifier %s\n" % (
        metrics.classification_report(teY, melt.classifyByThread(predicts, thread)))

    for start, end in zip(range(0, num_train_instances, batch_size),
                          range(batch_size, num_train_instances, batch_size)):
        trX, trY = trainset.mini_batch(start, end)
        sess.run(train_op, feed_dict=trainer.gen_feed_dict(trX, trY))

predicts, cost_ = sess.run([predict_op, cost], feed_dict=trainer.gen_feed_dict(teX, teY))
print 'final ', 'auc:', roc_auc_score(teY, predicts), 'cost:', cost_ / len(teY)
print "Classification report for classifier %s\n" % (
    metrics.classification_report(teY, melt.classifyByThread(predicts, thread)))
print "Confusion matrix:\n%s" % metrics.confusion_matrix(teY, melt.classifyByThread(predicts, thread))

if modelPath != '':
    saver = tf.train.Saver()
    save_path = saver.save(sess, modelPath)
    print("Model saved in file: %s" % save_path)