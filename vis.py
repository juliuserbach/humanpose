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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data import meta_info


def make_image(image):
    '''
    Helper function for display images in tensorboard. Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    :param image: numpy ndarray for the image
    :return: tf image summary for display in tensorboard
    '''

    from PIL import Image
    height, width, channel = image.shape
    image = Image.fromarray(image)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

def axisEqual3D(ax):
    '''
    Helper function for display 3D skeleton. Make the axis aspect ratio equal.
    :param ax: matplotlib axis containing 3D skeleton.
    '''
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def fig2rgb_array(fig, expand=False):
    '''
    Helper function for display 3D skeleton. Convert matplotlib figure to numpy array.
    :param fig: matplotlib fig.
    :return: corresponding numpy array.
    '''
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def display_image(image):
    '''
    Convert numpy array to tf image summary for image display in tensorboard.
    :param image: numpy array
    :return: tf image summary
    '''

    assert(image.ndim == 3)
    image = (image+1)*128
    image = image.astype(np.uint8)
    return make_image(image)

def display_pose3d(pose3d, format='H36'):
    '''
    Plot 3D joints positions in Matplotlib and convert to tf image summary for display in tensorboard.
    :param pose3d: numpy array of 3D joints [17,3] or [51]
    :param format: pose configuration [H36 or MPII]
    :return: tf image summary
    '''

    pose3d = np.reshape( pose3d, (17, -1) )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    limbs = meta_info.H36_LIMBS if format=='H36' else meta_info.MPII_LIMBS
    for limb in limbs:
        x, y, z = [np.array( [pose3d[limb[0], j], pose3d[limb[1], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, c='r' if limb[2] else 'b')
    ax.scatter(pose3d[:, 0],pose3d[:, 1],pose3d[:, 2],c='k')

    ax.view_init(elev=-90., azim=-90)

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])
    axisEqual3D(ax)

    image = make_image(fig2rgb_array(fig))
    plt.close(fig)
    return image

def display_pose2d(pose2d,image=None, format='H36'):
    '''
    Plot 2D joint positions on image or white background and convert to tf image summary for display in tensorboard.
    :param pose2d: numpy array of 3D joints [17,2] or [34]
    :param image: numpy array of person image
    :param format: pose configuration [H36 or MPII]
    :return: tf image summary
    '''
    pose2d = pose2d.astype(int)
    pose2d = np.reshape( pose2d, (-1, 2) )

    if image is None:
        x_max, y_max = pose2d[:,0].max(),pose2d[:,1].max()
        image = 255*np.ones((y_max,x_max,3),dtype=np.uint8)
    else:
        image = (image + 1) * 128
        image = image.astype(np.uint8)

    import cv2
    limbs = meta_info.H36_LIMBS if format=='H36' else meta_info.MPII_LIMBS
    for limb in limbs:
        x1,y1 = pose2d[limb[0],0],pose2d[limb[0],1]
        x2,y2 = pose2d[limb[1],0],pose2d[limb[1],1]
        image = cv2.line(image,(x1,y1),(x2,y2),color=(255,0,0) if limb[2] else (0,0,255),thickness=2)
    for joint in pose2d:
        x1,y1 = joint[0],joint[1]
        image = cv2.circle(image,(x1,y1),2, color=(0,0,0),thickness=-1)

    return make_image(image)

def log(tag, step, writer, image=None,value=None):
    '''
    Log image or value to tensorboard.
    :param tag: name tag for the data
    :param step: global training iteration
    :param writer: tf summary write
    :param image: tf image summary
    :param value: int or float value.
    '''
    if image is None:
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]),global_step=step)
    else:
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, image=image)]),global_step=step)