"""Copyright (c) 2020 AIT Lab, ETH Zurich

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""

import tensorflow as tf
import resnet_model
from tqdm import trange
from data import create_test_dataloader,meta_info
from utils import unnormalize_pose, generate_submission,create_zip_code_files
import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default='/cluster/project/infk/hilliges/lectures/mp20/project2/',help='path to the dataset')
parser.add_argument('--batch_size', type=int, default=1,help='batch size')
parser.add_argument('--log_dir', type=str, default='./example',help='log storage dir for tensorboard')
opt = parser.parse_args()

with tf.Session() as sess:

    # define resnet model
    sample = create_test_dataloader(data_root=opt.data_root, batch_size=opt.batch_size)
    with tf.variable_scope('model'):
        model = resnet_model.Model()
        p3d_out_norm = model(sample['image'], training=False)
    p3d_out = unnormalize_pose(p3d_out_norm)
    p3d_out = tf.reshape(p3d_out,[-1,51])

    # restore weights
    saver = tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint(opt.log_dir))

    predictions = None
    with trange(math.ceil(meta_info.NUM_SAMPLES_TEST/opt.batch_size)) as t:
        for i in t:
            p3d_out_ = sess.run(p3d_out)

            if predictions is None:
                predictions = p3d_out_
            else:
                predictions = np.concatenate([predictions,p3d_out_],axis=0)

    generate_submission(predictions, 'submission.csv.gz')
    create_zip_code_files('code.zip')
