'''

Define :
    checkpoint_root : Dir to save checkpoint
    summaries_root  : Dir to save summaries
    max_num_checkpoint : Max number of keeping checkpoint
    
    num_classes : Number of classes
    batch_size : Batch size
    num_epochs : Train epochs
    initial_learning_rate : Learning rate
    decay_factor : Learning rate decay factor
    num_epochs_per_decay : Epochs per decay

    fine_tuning : Load weights to train
    online_test : Test when train
    allow_soft_placement : Auto assign CPU or GPU
    log_device_placement : Log device info

'''
import os
import tensorflow as tf

flags = tf.app.flags


flags.DEFINE_string(
    'checkpoint_root', os.path.join(os.path.dirname(os.path.abspath(__file__)),'checkpoints'), 'CHECKPOINT DIR'
)

flags.DEFINE_string(
    'summaries_root', os.path.join(os.path.dirname(os.path.abspath(__file__)),'summaries'), 'SUMMARIES DIR'
)

flags.DEFINE_integer(
    'max_num_checkpoint', 10, 'MAX NUM CHECKPOINT'
)

flags.DEFINE_integer(
    'num_classes', 5, 'NUM OF CLASSES'
)

flags.DEFINE_integer(
    'batch_size', 50, 'BATCH SIZE'
)

flags.DEFINE_integer(
    'num_epochs', 10, 'EPOCHS'
)

flags.DEFINE_float(
    'initial_learning_rate', 0.1, 'INITIAL LEARNING RATE'
)

flags.DEFINE_float(
    'decay_factor', 0.95, 'LEARNING RATE DECAY'
)

flags.DEFINE_float(
    'num_epochs_per_decay', 1, 'NUM OF EPOCHS IN A DECAY'
)


flags.DEFINE_boolean(
    'fine_tuning', False, 'LOAD MODEL'
)

flags.DEFINE_boolean(
    'online_test', True, 'test when train'
)

flags.DEFINE_boolean(
    'allow_soft_placement', True, 'AUTO CPU OR GPU'
)

flags.DEFINE_boolean(
    'log_device_placement', False, 'LOG DEVICE INFO'
)

FLAGS = flags.FLAGS