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
import patoolib
import os
from data import meta_info

def normalize_pose(p3d):
    '''
    Normalize 3D joints.
    :param p3d: actual 3D joint posistions in shape [N_Batch, 17, 3]
    :return p3d: normalized 3D joint poisitions in shape [N_Batch, 17, 3]
    '''
    b = tf.shape(p3d)[0]
    p3d_mean_tf = tf.tile(tf.constant(meta_info.H36_POSE3D_MEAN), [b, 1, 1])
    p3d_std_tf = tf.tile(tf.constant(meta_info.H36_POSE3D_STD), [b, 1, 1])
    pelvis = tf.tile( tf.expand_dims(p3d[:,0,:],axis=1), [1,17,1])
    p3d = p3d - pelvis
    p3d = (p3d-p3d_mean_tf)/ p3d_std_tf
    p3d = tf.reshape(p3d, [-1, 51])
    return p3d

def unnormalize_pose(p3d):
    '''
    Normalize 3D joints.
    :param p3d: normalized 3D joint posistions in shape [N_Batch, 17, 3]
    :return p3d: actual 3D joint poisitions in shape [N_Batch, 17, 3]
    '''
    b = tf.shape(p3d)[0]
    p3d_mean_tf = tf.tile(tf.constant(meta_info.H36_POSE3D_MEAN), [b, 1, 1])
    p3d_std_tf = tf.tile(tf.constant(meta_info.H36_POSE3D_STD), [b, 1, 1])
    p3d = tf.reshape(p3d, [-1, 17, 3])
    pelvis = tf.tile(p3d[:,:1,:],[1,17,1])
    p3d = ( p3d - pelvis) * p3d_std_tf + p3d_mean_tf
    return p3d

def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def align_by_pelvis(joints):
    pelvis = joints[0, :]
    return joints - np.tile(np.expand_dims(pelvis, axis=0),[17,1])

def compute_errors(gt3ds, preds):
    '''
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    :param gt3ds: ground truth 3D joint poisitions in shape [N_Batch, 17, 3]
    :param preds: predicted 3D joint poisitions in shape [N_Batch, 17, 3]
    :return errors, errors_pa: MPJPE and MPJPE after Procrustes Analysis
    '''

    errors, errors_pa = [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        errors.append(np.mean(joint_error))

        # Get PA error.
        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa.append(np.mean(pa_error))

    errors = np.array(errors).mean()
    errors_pa = np.array(errors_pa).mean()

    return errors, errors_pa

def generate_submission(predictions, out_path):
    '''
    Generate result file for submission.
    :param predictions: predicted 3D joints in shape [N_Batch, 51]
    :param out_path: path to store the result file
    '''
    ids = np.arange(1, predictions.shape[0] + 1).reshape([-1, 1])

    predictions = np.hstack([ids, predictions])

    joints = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax', 'Neck/Nose', 'Head',
              'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
    header = ["Id"]

    for j in joints:
        header.append(j + "_x")
        header.append(j + "_y")
        header.append(j + "_z")

    header = ",".join(header)
    np.savetxt(out_path, predictions, delimiter=',', header=header, comments='')


code_files = [
    "data.py",
    "resnet_model.py",
    "train.py",
    "test.py",
    "utils.py",
    "vis.py"
]

def create_zip_code_files(out_path):
    '''
    Zip codes for submission.
    :param out_path: path to store the result file
    '''
    patoolib.create_archive(out_path, code_files)
