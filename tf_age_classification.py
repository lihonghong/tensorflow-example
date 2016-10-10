#!/usr/bin/env python
# coding=gbk
# ==============================================================================
# run :
# python ./tf_age_classification.py --train sex.train --test sex.validation
# ==============================================================================

import tensorflow as tf
# from sklearn import metrics
import melt
import model
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 60, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 500, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train', './corpus/feature.normed.rand.12000.0_2.txt', 'train file')
flags.DEFINE_string('test', './corpus/feature.normed.rand.12000.1_2.txt', 'test file')
flags.DEFINE_string('method', 'logistic', 'currently support logistic/mlp')
flags.DEFINE_integer('hidden_size', 50, 'Hidden unit size')
flags.DEFINE_integer('num_class', 7, 'output unit size')
flags.DEFINE_integer('gpu', 0, 'use which gpu')
flags.DEFINE_string('modelPath', '', 'path of saving train model')
flags.DEFINE_string("optimizer", "sgd", "optimizer to train: sgd,momentum,adadelta,adagrad,adam,ftrl,rmsprop")
flags.DEFINE_string("predictPath", '', "path of saving predict data")

trainset_file = FLAGS.train
testset_file = FLAGS.test

learning_rate = FLAGS.learning_rate
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
method = FLAGS.method
numClass = FLAGS.num_class
gpu = FLAGS.gpu

modelPath = FLAGS.modelPath
predictPath = FLAGS.predictPath

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

trainer = melt.gen_multi_classification_trainer(trainset, numClass)

py_x = model.Mlp().forward(trainer, FLAGS, numClass, gpu)
Y = trainer.Y

cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(py_x, Y))
train_op = model.getOptimizer(FLAGS, learning_rate).minimize(cost)  # construct optimizer
predict_op = tf.nn.sigmoid(py_x)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
init = tf.initialize_all_variables()
sess.run(init)

teX, teY = testset.full_batch()
teEncodeY = melt.oneHotLabel(teY)

num_train_instances = trainset.num_instances()
for i in range(num_epochs):
    predicts, cost_ = sess.run([predict_op, cost], feed_dict=trainer.gen_feed_dict(teX, teEncodeY))
    print i, 'cost:', cost_ / len(teY)
    # print "Classification report for classifier %s\n" % (metrics.classification_report(teY, melt.prob2Class(predicts)))

    for start, end in zip(range(0, num_train_instances, batch_size),
                          range(batch_size, num_train_instances, batch_size)):
        trX, trY = trainset.mini_batch(start, end)
        trY = melt.oneHotLabel(trY)
        sess.run(train_op, feed_dict=trainer.gen_feed_dict(trX, trY))

predicts, cost_ = sess.run([predict_op, cost], feed_dict=trainer.gen_feed_dict(teX, teEncodeY))
print 'final cost:', cost_ / len(teY)
# print "Classification report for classifier %s\n" % (metrics.classification_report(teY, melt.prob2Class(predicts)))
# print "Confusion matrix:\n%s" % metrics.confusion_matrix(teY, melt.prob2Class(predicts))

if modelPath != '':
    saver = tf.train.Saver()
    save_path = saver.save(sess, modelPath)
    print("Model saved in file: %s" % save_path)

if predictPath != '':
    np.savetxt(predictPath, np.c_[teY, predicts], delimiter=',', comments='',
               fmt='%d,%f,%f,%f,%f,%f,%f,%f')
    print("Predict result saved in :%s", predictPath)
