import melt
import tensorflow as tf


class LogisticRegresssion:
    def model(self, X, w):
        return melt.matmul(X, w)

    def forward(self, trainer):
        w = melt.init_weights([trainer.num_features, 1])
        py_x = self.model(trainer.X, w)
        return py_x


class Mlp:
    def model(self, X, w_h, w_o, gpu):
        with tf.device('/gpu:%d' % gpu):
            h = tf.nn.sigmoid(melt.matmul(X, w_h))  # this is a basic mlp, think 2 stacked logistic regressions
            return tf.matmul(h,
                             w_o)  # note that we dont take the softmax at the end because our cost fn does that for us

    def forward(self, trainer, FLAGS, numClass, gpu):
        w_h = melt.init_weights([trainer.num_features, FLAGS.hidden_size])  # create symbolic variables
        w_o = melt.init_weights([FLAGS.hidden_size, numClass])

        py_x = self.model(trainer.X, w_h, w_o, gpu)
        return py_x


class MlpBias:
    def model(self, X, w_h, b_h, w_o, b_o, gpu):
        with tf.device('/gpu:%d' % gpu):
            h = tf.nn.sigmoid(melt.matmul(X, w_h) + b_h)  # this is a basic mlp, think 2 stacked logistic regressions
            return tf.matmul(h,
                             w_o) + b_o  # note that we dont take the softmax at the end because our cost fn does that for us

    def forward(self, trainer, FLAGS, numClass, gpu):
        w_h = melt.init_weights([trainer.num_features, FLAGS.hidden_size])  # create symbolic variables
        b_h = melt.init_weights([FLAGS.hidden_size])  # create symbolic variables
        w_o = melt.init_weights([FLAGS.hidden_size, numClass])
        b_o = melt.init_weights([numClass])

        py_x = self.model(trainer.X, w_h, b_h, w_o, b_o, gpu)
        return py_x


class Mlp2Layer:
    def model(self, X, w_h1, w_h2, w_o):
        h1 = tf.nn.sigmoid(melt.matmul(X, w_h1))  # this is a basic mlp, think 2 stacked logistic regressions
        h2 = tf.nn.sigmoid(melt.matmul(h1, w_h2))  # this is a basic mlp, think 2 stacked logistic regressions
        return tf.matmul(h2, w_o)  # note that we dont take the softmax at the end because our cost fn does that for us

    def forward(self, trainer, FLAGS, numClass):
        w_h1 = melt.init_weights([trainer.num_features, FLAGS.hidden_size])  # create symbolic variables
        w_h2 = melt.init_weights([FLAGS.hidden_size, FLAGS.hidden_size])  # create symbolic variables
        w_o = melt.init_weights([FLAGS.hidden_size, numClass])

        py_x = self.model(trainer.X, w_h1, w_h2, w_o)
        return py_x


def getOptimizer(FLAGS, learning_rate):
    if FLAGS.optimizer == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif FLAGS.optimizer == "momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate)
    elif FLAGS.optimizer == "adadelta":
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif FLAGS.optimizer == "adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif FLAGS.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif FLAGS.optimizer == "ftrl":
        optimizer = tf.train.FtrlOptimizer(learning_rate)
    elif FLAGS.optimizer == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
        print("Unknow optimizer: {}, exit now".format(FLAGS.optimizer))
        exit(1)
    return optimizer
