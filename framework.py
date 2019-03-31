import os
import sys
import tensorflow as tf
import numpy as np
from flags import FLAGS


class Framework:
    def __init__(self, train_data_loader, test_data_loader):
        self.train_data_loader = train_data_loader
        self.test_data_loader = train_data_loader
        self.num_features = self.train_data_loader.num_features
        self.sess_config = tf.ConfigProto(
            allow_soft_placement = FLAGS.allow_soft_placement,
            log_device_placement = FLAGS.log_device_placement
        )
        self.sess = None
        self.graph = None

    def build(self, model, prefix):
        self.graph = tf.Graph()
        self.prefix = prefix
        with self.graph.as_default():
            self.add_global_step()
            self.add_learning_rate()
            self.add_placeholder()
            self.logits = model(self.placeholders)
            self.add_loss()
            self.add_accuracy()
            self.add_train_op()
            self.add_summary()
            self.train_summary_writer.add_graph(self.graph)
            self.test_summary_writer.add_graph(self.graph)
            self.saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoint)
              
    def add_global_step(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
    def add_train_op(self):
        raise NotImplementedError
    def add_placeholder(self):
        raise NotImplementedError

    def add_learning_rate(self):
        raise NotImplementedError
    def add_summary(self):
        tf.summary.scalar("loss", self.loss, collections=['train','test'])
        tf.summary.scalar("accuracy", self.accuracy, collections=['train', 'test'])
        tf.summary.scalar("global_step", self.global_step, collections=['train'])
        tf.summary.scalar("learning_rate", self.learning_rate, collections=['train'])

        self.summary_train_op = tf.summary.merge_all('train')
        self.summary_test_op = tf.summary.merge_all("test")
        train_summary_folder = os.path.join(FLAGS.summaries_root, self.prefix, 'train')
        self.train_summary_writer = tf.summary.FileWriter(train_summary_folder)
        test_summary_folder = os.path.join(FLAGS.summaries_root, self.prefix, 'test')
        self.test_summary_writer = tf.summary.FileWriter(test_summary_folder)
        
        
        
    def add_loss(self):
        raise NotImplementedError

    def add_accuracy(self):
        raise NotImplementedError

    def get_feed_dict(self, batch_data):
        raise NotImplementedError
    def train(self, model, prefix='default'):
        if self.graph is None:
            self.build(model,prefix)
            print(self.prefix)
        self.sess = tf.Session(graph=self.graph, config=self.sess_config)
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())

            if FLAGS.fine_tuning:
                self.saver.restore(sess, os.path.join(FLAGS.checkpoint_root, self.prefix))

            for epoch in range(FLAGS.num_epochs):
                for idx, batch_data in enumerate(self.train_data_loader):

                    batch_loss, _, train_summaries, train_step = sess.run(
                        [self.loss, self.train_op, self.summary_train_op, self.global_step],
                        feed_dict = self.get_feed_dict(batch_data)
                    )

                    self.train_summary_writer.add_summary(train_summaries, global_step=train_step)
                    sys.stdout.write('Epoch: {} | step {} | Loss: {:.3f}\r'.format(epoch+1,idx+1, batch_loss))
                    sys.stdout.flush()
                if FLAGS.online_test:
                    self.test(model)
                
            if not os.path.exists(FLAGS.checkpoint_root):
                os.makedirs(FLAGS.checkpoint_root)
            path = self.saver.save(sess, os.path.join(FLAGS.checkpoint_root, self.prefix))
            print("\nModel saved in {}".format(
                path
            ))
            
    def test(self, model, prefix=None):
        if self.graph is None:
            if prefix is None:
                raise ValueError("...what you want to do?")
            self.build(model, prefix)
        if self.sess is None:
            self.sess = tf.Session(graph=self.graph, config=self.sess_config)
            self.sess.run(tf.global_variables_initializer())
        with self.sess.as_default() as sess:
            if prefix is not None:
                self.saver.restore(sess, os.path.join(FLAGS.checkpoint_root, self.prefix))
            accuracy = []
            print()
            for idx, batch_data in enumerate(self.test_data_loader):
                test_accuracy, test_summaries = sess.run(
                    [self.accuracy, self.summary_test_op],
                    feed_dict=self.get_feed_dict(batch_data)
                )
                current_step = tf.train.global_step(sess, self.global_step)
                
                accuracy.append(test_accuracy)
                sys.stdout.write("[TEST] step {} | accuracy: {:.3f}\r".format(idx+1, test_accuracy))
                sys.stdout.flush()
            self.test_summary_writer.add_summary(test_summaries, global_step=current_step)
            print("\nFinal accuracy: {:.3f}".format(np.mean(accuracy)))


class UserFramework(Framework):

    def add_placeholder(self):
        self.placeholders = {
            "inputs" : tf.placeholder(tf.float32, shape=([None, self.num_features]), name='inputs'),
            "labels" : tf.placeholder(tf.int32, shape=([None,]), name='labels')
        }
        self.labels_one_hot = tf.one_hot(self.placeholders['labels'], depth= FLAGS.num_classes, axis=-1)

    def add_learning_rate(self):
        decay_steps = int(self.train_data_loader.num_samples / FLAGS.batch_size * FLAGS.num_epochs_per_decay)
        self.learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
            self.global_step, decay_steps, FLAGS.decay_factor, staircase=True, name='exp_leaning_rate_decay'
            )
    
    def add_loss(self):
        self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels_one_hot)
                    )
    def add_accuracy(self):
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(
                tf.argmax(self.logits, axis=1), tf.argmax(self.labels_one_hot, axis=1)
                ), tf.float32)
        )

    def add_train_op(self):
        with tf.name_scope("train_op"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            grads = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)

    def get_feed_dict(self, batch_data):
        feed_dict = {
            self.placeholders['inputs'] : batch_data['inputs'],
            self.placeholders['labels'] : batch_data['labels']
        }
        return feed_dict


