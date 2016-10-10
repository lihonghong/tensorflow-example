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
    def model(self, X, w_h, w_o):
        h = tf.nn.sigmoid(melt.matmul(X, w_h))  # this is a basic mlp, think 2 stacked logistic regressions
        return tf.matmul(h, w_o)  # note that we dont take the softmax at the end because our cost fn does that for us

    def forward(self, trainer, FLAGS, numClass):
        w_h = melt.init_weights([trainer.num_features, FLAGS.hidden_size])  # create symbolic variables
        w_o = melt.init_weights([FLAGS.hidden_size, numClass])

        py_x = self.model(trainer.X, w_h, w_o)
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
