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

class meta_info():
    # Number of samples in each dataset
    NUM_SAMPLES_H36 = 636724
    NUM_SAMPLES_TEST = 2181
    NUM_SAMPLES_MPI = 18000

    # H36 joint names
    H36M_NAMES = [''] * 17
    H36M_NAMES[0] = 'Hip'
    H36M_NAMES[1] = 'RHip'
    H36M_NAMES[2] = 'RKnee'
    H36M_NAMES[3] = 'RFoot'
    H36M_NAMES[4] = 'LHip'
    H36M_NAMES[5] = 'LKnee'
    H36M_NAMES[6] = 'LFoot'
    H36M_NAMES[7] = 'Spine'
    H36M_NAMES[8] = 'Thorax'
    H36M_NAMES[9] = 'Neck/Nose'
    H36M_NAMES[10] = 'Head'
    H36M_NAMES[11] = 'LShoulder'
    H36M_NAMES[12] = 'LElbow'
    H36M_NAMES[13] = 'LWrist'
    H36M_NAMES[14] = 'RShoulder'
    H36M_NAMES[15] = 'RElbow'
    H36M_NAMES[16] = 'RWrist'

    # MPII joint names
    MPII_NAMES = [''] * 16
    MPII_NAMES[0] = 'RAnkle'
    MPII_NAMES[1] = 'RKnee'
    MPII_NAMES[2] = 'RHip'
    MPII_NAMES[3] = 'LHip'
    MPII_NAMES[4] = 'LKnee'
    MPII_NAMES[5] = 'LAnkle'
    MPII_NAMES[6] = 'Pelvis'
    MPII_NAMES[7] = 'Thorax'
    MPII_NAMES[8] = 'Neck'
    MPII_NAMES[9] = 'Head'
    MPII_NAMES[10] = 'RWrist'
    MPII_NAMES[11] = 'RElbow'
    MPII_NAMES[12] = 'RShoulder'
    MPII_NAMES[13] = 'LShoulder'
    MPII_NAMES[14] = 'LElbow'
    MPII_NAMES[15] = 'LWrist'

    # Limb definitions: each entry: [joint1,joint2,right (1) or left(0)]
    H36_LIMBS = [[0, 1, 1], [1, 2, 1], [2, 3, 1], [0, 4, 0], [4, 5, 0], [5, 6, 0], [0, 7, 0],
                 [7, 8, 0], [8, 9, 0], [9, 10, 0], [8, 11, 0], [11, 12, 0], [12, 13, 0],
                 [8, 14, 1], [14, 15, 1], [15, 16, 1]]

    MPII_LIMBS = [[0, 1, 1], [1, 2, 1], [3, 4, 0], [4, 5, 0], [2, 6, 1], [3, 6, 0], [6, 7, 0],
                  [7, 8, 0], [8, 9, 0], [7, 12, 1], [7, 13, 0], [12, 11, 1], [11, 10, 1],
                  [13, 14, 0], [14, 15, 0]]

    # Read mean and std for data normalization
    H36_POSE3D_MEAN = np.load(os.path.join('./', 'misc', "mean.npy")).reshape([1, 17, 3]).astype(np.float32)
    H36_POSE3D_STD = np.load(os.path.join('./', 'misc', "std.npy")).reshape([1, 17, 3]).astype(np.float32)
    # Pelvis STD is zero, causing numerical error. Set to 1 to prevent this issue.
    # Before normalization and un-normalization, pelvis coordniates need to be substrcated from all joints.
    H36_POSE3D_STD[:, 0, :] = 1

def create_H36_dataloader(data_root, batch_size, subjects):

    tf_data_files = tf.data.Dataset.list_files(os.path.join(data_root,'h36','*'), shuffle=True)
    dataset = tf.data.TFRecordDataset(filenames=tf_data_files, compression_type="ZLIB",num_parallel_reads=4)

    # Parse tf example
    def _parse(example_proto):

        # Disribe all features stored in the tfrecords. 'image', 'pose2d_crop', 'pose3d' and 'subject' are particularly relevent for the project.
        image_feature_description = {
            # *image : images in size 256x256, cropped from the original images and scaled. Human bodies are located at the image center.
            'image': tf.io.FixedLenFeature([], tf.string),
            # offset : parameter for cropping. Each pixel in image corresponds to [pixel / scale + offset in the original image].
            'offset': tf.io.FixedLenFeature([], tf.string),
            # scale : parameter for cropping. Each pixel in image corresponds to [pixel / scale + offset in the original image].
            'scale': tf.io.FixedLenFeature([], tf.float32),
            # *pose2d_crop : 2D pixel coordinates of 17 joints in cropped images. [pose2d_crop = (pose2d - offset) * scale]
            'pose2d_crop': tf.io.FixedLenFeature([], tf.string),
            # pose2d : 2D pixel coordinates of 17 joints in original images. [pose2d = intrinsics * pose3d]
            'pose2d': tf.io.FixedLenFeature([], tf.string),
            # *pose3d :  3D positions of 17 joints in original camera frames. [pose2d = intrinsics * pose3d]
            'pose3d': tf.io.FixedLenFeature([], tf.string),
            # pose3d_univ :  3D positions of 17 joints in a canonical world coordiante frame (from MoCap). [pose2d = intrinsics_univ * pose3d_univ]
            'pose3d_univ': tf.io.FixedLenFeature([], tf.string),
            # intrinsics : matrix of the camera for projecting pose3d to pose2d. [pose2d = intrinsics * pose3d]
            'intrinsics': tf.io.FixedLenFeature([], tf.string),
            # intrinsics_univ : matrix for projecting pose3d_univ to pose2d.. [pose2d = intrinsics_univ * pose3d_univ]
            'intrinsics_univ': tf.io.FixedLenFeature([], tf.string),
            # framd_id : frame ID.
            'framd_id': tf.io.FixedLenFeature([], tf.int64),
            # camera : The data is capture using 4 cameras. This is the camera ID.
            'camera': tf.io.FixedLenFeature([], tf.int64),
            # *subject : ID of the person in the image. [1,5,6,7,8]
            'subject': tf.io.FixedLenFeature([], tf.int64),
            # action : ID of the action performed in the image.  2-Directions,3-Discussion, 4-Eating, 5-Greeting, 6-Phoning, 7-Posing, 8-Purchases, 9-Sitting, 10-Sitting Down, 11-Smoking, 12-Taking Photo, 13-Waiting, 14-Walking, 15-Walking Dog, 16-Walk Together
            'action': tf.io.FixedLenFeature([], tf.int64),
            # subaction : Each action contains several sequences. This is the sequence ID.
            'subaction':  tf.io.FixedLenFeature([], tf.int64),
        }

        sample = tf.io.parse_single_example(example_proto, image_feature_description)

        sample['pose3d'] = tf.decode_raw(sample['pose3d'], tf.float32)
        sample['pose3d'] = tf.reshape(sample['pose3d'], (17, 3))

        sample['pose2d_crop'] = tf.decode_raw(sample['pose2d_crop'], tf.float32)
        sample['pose2d_crop'] = tf.reshape(sample['pose2d_crop'], (17, 2))

        sample['image'] = tf.decode_raw(sample['image'], tf.uint8)
        sample['image'] = tf.reshape(sample['image'], (256, 256, 3))

        return sample

    dataset = dataset.map(_parse,num_parallel_calls=4)

    # Take requested subjects only
    def _filter(sample):
        flag = tf.count_nonzero(tf.equal(sample['subject'], tf.constant(subjects,dtype=tf.int64)))
        return tf.cast(flag,tf.bool)

    dataset = dataset.filter(_filter)

    # Preprocess image and pose (for augmentation)
    def _preprocess(sample):

        sample['image'] = tf.image.resize_images(sample['image'], (224, 224))
        sample['image'] = tf.cast(sample['image'], tf.float32) / 128. - 1

        sample['pose2d_crop'] = sample['pose2d_crop'] * 224./255.

        return sample
    dataset = dataset.map(_preprocess,num_parallel_calls=4)

    dataset= dataset.prefetch(batch_size * 10)
    dataset = dataset.shuffle(batch_size * 10,reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset= dataset.prefetch(2)

    iterator = dataset.make_one_shot_iterator()
    dataloader = iterator.get_next()

    return dataloader

def create_test_dataloader(data_root, batch_size):
    dataset = tf.data.TFRecordDataset(filenames=os.path.join(data_root,'test_set.tfrecords'),
                                      compression_type="ZLIB")

    # Parse tf example
    def _parse(example_proto):
        image_feature_description = {
            'image':
                tf.io.FixedLenFeature([], tf.string),
            'offset':
                tf.io.FixedLenFeature([], tf.string),
            'scale':
                tf.io.FixedLenFeature([], tf.float32),
            'intrinsics':
                tf.io.FixedLenFeature([], tf.string),
            'intrinsics_univ':
                tf.io.FixedLenFeature([], tf.string),
            'camera':
                tf.io.FixedLenFeature([], tf.int64),
            'subject':
                tf.io.FixedLenFeature([], tf.int64),
            'action':
                tf.io.FixedLenFeature([], tf.int64),
            'subaction':
                tf.io.FixedLenFeature([], tf.int64),
        }

        sample = tf.io.parse_single_example(example_proto, image_feature_description)

        sample['image'] = tf.decode_raw(sample['image'], tf.uint8)
        sample['image'] = tf.reshape(sample['image'], (256, 256, 3))

        return sample

    dataset = dataset.map(_parse)

    # Preproces image and pose (for augmentation)
    def _preprocess(sample):
        sample['image'] = tf.image.resize_images(sample['image'], (224, 224))
        sample['image'] = tf.cast(sample['image'], tf.float32) / 128. - 1
        return sample

    dataset = dataset.map(_preprocess)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataloader = dataset.make_one_shot_iterator().get_next()

    return dataloader


def create_MPII_dataloader(data_root, batch_size):
    tf_data_files = tf.data.Dataset.list_files(os.path.join(data_root,'mpii','*'), shuffle=True)
    dataset = tf.data.TFRecordDataset(filenames=tf_data_files)

    def _parse(example_proto):
        feature_map = {
            # encoded : encoded images which can be decoded to raw image. See code below.
            'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            # pose2d : 2D pixel coordinates of joints.
            'pose2d': tf.FixedLenFeature([], dtype=tf.string),
            # visibility : 2D joint visibility. 0 means the joint is not visible hence the joint position annotation is invalid.
            'visibility': tf.FixedLenFeature((1, 16), dtype=tf.int64),
        }

        sample = tf.parse_single_example(example_proto, feature_map)

        sample['image'] = tf.image.decode_jpeg(sample['image'], channels=3)

        sample['pose2d'] = tf.decode_raw(sample['pose2d'], tf.float32)
        sample['pose2d'] = tf.reshape(sample['pose2d'], (16, 2))

        sample['vis'] = tf.cast(sample['visibility'], dtype=tf.float32)

        return sample

    dataset = dataset.map(_parse,num_parallel_calls=4)

    def _preprocess(sample):
        sample['image'] = tf.cast(sample['image'], tf.float32) / 128. - 1
        return sample
    dataset = dataset.map(_preprocess,num_parallel_calls=4)

    dataset= dataset.prefetch(batch_size * 10)
    dataset = dataset.shuffle(batch_size * 10,reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset= dataset.prefetch(2)

    dataloader = dataset.make_one_shot_iterator().get_next()
    return dataloader

