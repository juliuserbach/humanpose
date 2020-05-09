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
import numpy as np
import os

from utils import normalize_pose,unnormalize_pose,compute_errors
from vis import display_pose3d,display_image,log
from data import create_H36_dataloader,meta_info
import resnet_model

# Parameters
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_root', type=str, default='/cluster/project/infk/hilliges/lectures/mp20/project2/',help='path to the dataset')
parser.add_argument('--n_epochs', type=int, default=5,help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=64,help='batch size')
parser.add_argument('--lr', type=float, default=0.01,help='learning rate')
parser.add_argument('--log_dir', type=str, default='./example',help='log storage dir for tensorboard')
parser.add_argument('--freq_log', type=int, default=10,help='frequency of logging training loss into tensorboard')
parser.add_argument('--freq_display', type=int, default=100,help='frequency of logging training loss into tensorboard')
parser.add_argument('--freq_save', type=int, default=1000,help='frequency of saving model')
parser.add_argument('--val_subject', type=int, default=-1,help='which subject for validation [1,5,6,7,8]. -1 if no validation ')
parser.add_argument('--freq_val', type=int, default=1000,help='frequency of validating model')
parser.add_argument('--val_steps', type=int, default=20,help='number of batches for validation')
opt = parser.parse_args()

TRAINING_SUBJECT = [1,5,6,7,8]

# Setup wandb for visualization
import wandb
wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)
wandb.config.update(opt)

# Config tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
with tf.Session(config=config) as sess:

    # Read data
    if opt.val_subject in TRAINING_SUBJECT:
        TRAINING_SUBJECT.remove(opt.val_subject)
    sample = create_H36_dataloader(data_root=opt.data_root, batch_size=opt.batch_size, subjects=TRAINING_SUBJECT)
    image,pose3d_gt = sample['image'],sample['pose3d']

    # Normalize pose
    pose3d_gt_norm = normalize_pose(pose3d_gt)

    # Predict pose
    with tf.variable_scope("model", reuse=False):
        model = resnet_model.Model()
        pose3d_out_norm = model(image, training=True)

    # Compare with GT
    loss = tf.losses.absolute_difference(pose3d_gt_norm, pose3d_out_norm)

    # Optimize network parameters
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=opt.lr).minimize(loss)

    # Unnormalize pose
    pose3d_out = unnormalize_pose(pose3d_out_norm)

    # Validation graph
    if not opt.val_subject == -1:
        sample_valid = create_H36_dataloader(data_root=opt.data_root, batch_size=opt.batch_size, subjects=[opt.val_subject])
        image_valid,pose3d_gt_valid = sample_valid['image'],sample_valid['pose3d']
        pose3d_gt_norm_valid = normalize_pose(pose3d_gt_valid)
        with tf.variable_scope("model", reuse=True):
            model_valid = resnet_model.Model()
            pose3d_out_norm_valid = model_valid(image_valid, training=False)
        loss_valid = tf.losses.absolute_difference(pose3d_gt_norm_valid, pose3d_out_norm_valid)
        pose3d_out_valid = unnormalize_pose(pose3d_out_norm_valid)

    # Define tensorboard writer
    summary_writer = tf.summary.FileWriter(opt.log_dir, sess.graph)

    # Initialize
    tf.global_variables_initializer().run()

    # Define model saver
    saver = tf.train.Saver(tf.global_variables())

    # Training loop
    from tqdm import trange
    with trange(int(opt.n_epochs * meta_info.NUM_SAMPLES_H36 / opt.batch_size)) as t:
        for i in t:

            # Run one train iteration
            _,pose3d_out_,pose3d_gt_,loss_,image_,pose2d_gt_ = sess.run([train_op,pose3d_out,pose3d_gt,loss,image,sample['pose2d_crop']])

            # Display training status
            epoch_cur = i * opt.batch_size// meta_info.NUM_SAMPLES_H36
            iter_cur = (i * opt.batch_size ) % meta_info.NUM_SAMPLES_H36
            t.set_postfix(epoch=epoch_cur,iter_percent="%d %%"%(iter_cur/float(meta_info.NUM_SAMPLES_H36)*100),loss='%.3f'%loss_)

            # Log numerical reuslts
            if i % opt.freq_log == 0:
                mpjpe_, pa_mpjpe_ = compute_errors(pose3d_out_,pose3d_gt_)
                log(tag='train/loss', step=i, writer=summary_writer, value=loss_)
                log(tag='train/mpjpe', step=i, writer=summary_writer, value=mpjpe_)
                log(tag='train/pa_mpjpe', step=i, writer=summary_writer, value=pa_mpjpe_)

            # Log visual reuslts
            if i % opt.freq_display == 0:
                log(tag='train/input_image', step=i, writer=summary_writer, image=display_image(image_[0]))
                log(tag='train/pose3d_pred', step=i, writer=summary_writer, image=display_pose3d(pose3d_out_[0]))
                log(tag='train/pose3d_gt', step=i, writer=summary_writer, image=display_pose3d(pose3d_gt_[0]))

            # Validate
            if i % opt.freq_val == 0 and i is not 0 and not opt.val_subject == -1:

                loss_valid_list, mpjpe_valid_list, pa_mpjpe_valid_list = [],[],[]
                for j in range(opt.val_steps):
                    pose3d_out_valid_, pose3d_gt_valid_, loss_valid_,image_valid_ = sess.run([pose3d_out_valid,pose3d_gt_valid,loss_valid,image_valid])
                    mpjpe_, pa_mpjpe_ = compute_errors(pose3d_out_valid_, pose3d_gt_valid_)
                    loss_valid_list.append(loss_valid_)
                    mpjpe_valid_list.append(mpjpe_)
                    pa_mpjpe_valid_list.append(pa_mpjpe_)

                log(tag='valid/loss', step=i, writer=summary_writer, value=np.array(loss_valid_list).mean())
                log(tag='valid/mpjpe', step=i, writer=summary_writer, value=np.array(mpjpe_valid_list).mean())
                log(tag='valid/pa_mpjpe', step=i, writer=summary_writer, value=np.array(pa_mpjpe_valid_list).mean())

                log(tag='valid/input_image', step=i, writer=summary_writer, image=display_image(image_valid_[0]))
                log(tag='valid/pose3d_pred', step=i, writer=summary_writer, image=display_pose3d(pose3d_out_valid_[0]))
                log(tag='valid/pose3d_gt', step=i, writer=summary_writer, image=display_pose3d(pose3d_gt_valid_[0]))

            # Save model
            if i % opt.freq_save == 0:
                saver.save(sess,os.path.join(opt.log_dir,"model"),global_step=i)

    saver.save(sess,os.path.join(opt.log_dir,"model"),global_step=int(opt.n_epochs * meta_info.NUM_SAMPLES_H36 / opt.batch_size))

