
'''
Author: Talip Ucar
Disclaimer: Some of the utility functions are taken from PrGAN github page: https://github.com/matheusgadelha/PrGAN
Some functions are used diretly, or used after modified from the original code.
'''
	

import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import sys
import re
from functools import reduce
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec

			


def flatten(t) : return tf.reshape(t, [1, -1])


def rot_matrix(s):

    theta = (s + 1.0) * np.pi
    phi = 0. #s[1] * np.pi/2.0

    sin_theta = tf.sin(theta)
    cos_theta = tf.cos(theta)
    sin_phi = tf.sin(phi)
    cos_phi = tf.cos(phi)

    ry = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
    rx = [[1, 0, 0], [0, cos_phi, -sin_phi], [0, sin_phi, cos_phi]]

    return tf.matmul(rx, ry)


'''This is a hack. One day tensorflow will have gather_nd properly implemented with backprop.
Until then, use this function'''
def gather_nd(params, indices, name=None):
    shape = params.get_shape().as_list()
    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x*y, shape[i+1:], 1) for i in range(0, rank)]
    indices_unpacked = tf.unstack(tf.transpose(indices, [rank - 1] + list(range(0, rank - 1)), name))
    flat_indices = sum([a*b for a,b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices, name=name)


def grid_coord(h, w, d):
    xl = tf.linspace(-1.0, 1.0, w)
    yl = tf.linspace(-1.0, 1.0, h)
    zl = tf.linspace(-1.0, 1.0, d)

    xs, ys, zs = tf.meshgrid(xl, yl, zl, indexing='ij')
    g = tf.concat([flatten(xs), flatten(ys), flatten(zs)], 0)
    return g


def project(v, tau=1):
    p = tf.reduce_sum(v, 2)
    p = tf.ones_like(p) - tf.exp(-p*tau)
    return tf.reverse(tf.transpose(p), [True, False])


def get_voxel_values(v, xs, ys, zs):
    idxs = tf.cast(tf.stack([xs, ys, zs], axis=1), 'int32')
    idxs = tf.clip_by_value(idxs, 0, v.get_shape()[0])
    idxs = tf.expand_dims(idxs, 0)
    return gather_nd(v, idxs)


def resample_voxels(v, xs, ys, zs, method="trilinear"):
    
    if method == "trilinear":
        floor_xs = tf.floor(tf.clip_by_value(xs, 0, 64))
        floor_ys = tf.floor(tf.clip_by_value(ys, 0, 64))
        floor_zs = tf.floor(tf.clip_by_value(zs, 0, 64))

        ceil_xs = tf.ceil(tf.clip_by_value(xs, 0, 64))
        ceil_ys = tf.ceil(tf.clip_by_value(ys, 0, 64))
        ceil_zs = tf.ceil(tf.clip_by_value(zs, 0, 64))

        final_value =( tf.abs((xs-floor_xs)*(ys-floor_ys)*(zs-floor_zs))*get_voxel_values(v, ceil_xs, ceil_ys, ceil_zs) + 
                       tf.abs((xs-floor_xs)*(ys-floor_ys)*(zs-ceil_zs))*get_voxel_values(v, ceil_xs, ceil_ys, floor_zs) +
                       tf.abs((xs-floor_xs)*(ys-ceil_ys)*(zs-floor_zs))*get_voxel_values(v, ceil_xs, floor_ys, ceil_zs) +
                       tf.abs((xs-floor_xs)*(ys-ceil_ys)*(zs-ceil_zs))*get_voxel_values(v, ceil_xs, floor_ys, floor_zs) +
                       tf.abs((xs-ceil_xs)*(ys-floor_ys)*(zs-floor_zs))*get_voxel_values(v, floor_xs, ceil_ys, ceil_zs) +
                       tf.abs((xs-ceil_xs)*(ys-floor_ys)*(zs-ceil_zs))*get_voxel_values(v, floor_xs, ceil_ys, floor_zs) +
                       tf.abs((xs-ceil_xs)*(ys-ceil_ys)*(zs-floor_zs))*get_voxel_values(v, floor_xs, floor_ys, ceil_zs) +
                       tf.abs((xs-ceil_xs)*(ys-ceil_ys)*(zs-ceil_zs))*get_voxel_values(v, floor_xs, floor_ys, floor_zs)
                     )
        return final_value
    
    elif method == "nearest":
        r_xs = tf.round(xs)
        r_ys = tf.round(ys)
        r_zs = tf.round(zs)
        return get_voxel_values(v, r_xs, r_ys, r_zs)
    
    else:
        raise NameError(method)
  
  
def transform_volume(v, t):
    height = int(v.get_shape()[0])
    width = int(v.get_shape()[1])
    depth = int(v.get_shape()[2])
    grid = grid_coord(height, width, depth)
    
    xs = grid[0, :]
    ys = grid[1, :]
    zs = grid[2, :]
    
    idxs_f = tf.transpose(tf.stack([xs, ys, zs]))
    idxs_f = tf.matmul(idxs_f, t)
    
    xs_t = (idxs_f[:, 0] + 1.0) * float(width) / 2.0
    ys_t = (idxs_f[:, 1] + 1.0) * float(height) / 2.0
    zs_t = (idxs_f[:, 2] + 1.0) * float(depth) / 2.0
    
    return tf.reshape(resample_voxels(v, xs_t, ys_t, zs_t, method='trilinear'), v.get_shape())





def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def linear(x, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        w = tf.get_variable(
            name='w',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer())
        h = tf.matmul(x, w) + b
        if activation is not None:
            h = activation(h)
        return h


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def conv3d(input_, output_dim, k_h=3, k_w=3, k_d=3, d_h=2, d_w=2, d_d=2, stddev=0.02,
           name="conv3d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_d, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


# From DCGAN tensrorflow implementation
def deconv3d(input_, output_shape,
             k_h=3, k_w=3, k_d=3,
             d_h=2, d_w=2, d_d=2,
             stddev=0.02,
             name='deconv3d',
             with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_d, k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape, strides=[1, d_d, d_h, d_w, 1])
        except AttributeError:
            print("This tensorflow version does not supprot tf.nn.conv3d_transpose.")

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def l2norm_sqrd(a, b): return tf.reduce_sum(tf.pow(a-b, 2), 1)


def l2(a, b): return tf.reduce_mean(tf.pow(a-b, 2))



def load_imgbatch(img_paths, color=True):
    images = []
    if color:
        for path in img_paths:
            images.append(mpimg.imread(path)[:, :, 0:3])
    else:
        for path in img_paths:
            img = mpimg.imread(path)
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            images.append(img)
    return images



def save_voxels(voxels, folder, epoch, mode="train"):
    basename="/"+str(epoch)+mode+"volume{}.npy"

    if epoch % 50 == 20:
        for i in range(voxels.shape[0]):
            np.save(folder+basename.format(i), voxels[i, :, :, :])
    else:
        i=0
        np.save(folder+basename.format(i), voxels[i, :, :, :])


def save_voxels_as_image(decoder_output, binvox_size, results_directory, i, elevation=30, azimuth=45, mode='train', threshold=0.5):
    r, g, b = np.indices((binvox_size, binvox_size, binvox_size)) / binvox_size
    voxel = decoder_output>= threshold
    voxel = voxel.astype(np.bool)
    fig = plt.figure(figsize=(8,5))
    ax = fig.gca(projection='3d')

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
    ax._axis3don = False

    colors = np.zeros(voxel.reshape(-1,binvox_size,binvox_size).shape + (3,))
    colors[..., 0] = r
    colors[..., 1] = g
    colors[..., 2] = b
    ax.voxels(np.transpose(voxel.reshape(-1,binvox_size,binvox_size),(2,0,1)), facecolors=colors, edgecolors=np.clip(2*colors - 0.5, 0, 1), linewidth=0.5)
  
    ax.view_init(elevation, azimuth) #(Elevation, Azimuth)
    save_file='{}{}.png'.format('Epoch_'+str(i)+'_'+mode+'_th'+str(threshold)+"_angle_", azimuth)
    plt.savefig(results_directory+save_file)
    
    if mode == 'sampled':
        for azimuth in range(0,360,15):
            ax.view_init(elevation, azimuth) #Elevation and Azimuth
            save_file='{}{}.png'.format('Gif_Epoch_'+str(i)+'_'+mode+"_"+str(threshold)+"_angle_", azimuth)
            plt.savefig(results_directory+save_file)

        plt.clf()

    elif i % 1000 == 900:
        for azimuth in range(0,360,15):
            ax.view_init(elevation, azimuth) #Elevation and Azimuth
            save_file='{}{}.png'.format('Gif_Epoch_'+str(i)+'_'+mode+"_"+str(threshold)+"_angle_", azimuth)
            plt.savefig(results_directory+save_file)
        plt.clf()

    else:
        print('Are you sure that you wanna plot voxels?')
  











def save_image_voxel_grid(image, decoder_output, binvox_size, results_directory, i, elevation=30, azimuth=45, mode='train', threshold=0.5):

    image = 255.*image

    save_file='{}.png'.format(mode+"_ImgIdx_"+str(i)+'_th_'+str(threshold))
    r, g, b = np.indices((binvox_size, binvox_size, binvox_size)) / binvox_size
    voxel = decoder_output>= threshold
    voxel = voxel.astype(np.bool)

    # Set up a figure 1/10 as tall as it is wide
    fig = plt.figure(figsize=plt.figaspect(1./9), frameon=False)


    ax = fig.add_subplot(1, 9, 1)

    # make the panes transparent
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    j=2
    for azimuth in range(0,360,45):
        ax = fig.add_subplot(1, 9, j, projection='3d')

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
        ax.axis('off')
        ax.set_xlim3d([0,binvox_size-1])
        ax.set_ylim3d([0,binvox_size-1])
        ax.set_zlim3d([0,binvox_size-1])

        fig.add_axes(ax)

        colors = np.zeros(voxel.reshape(-1,binvox_size,binvox_size).shape + (3,))
        colors[..., 0] = r
        colors[..., 1] = g
        colors[..., 2] = b
        ax.voxels(np.transpose(voxel.reshape(-1,binvox_size,binvox_size),(2,0,1)), facecolors=colors, edgecolors=np.clip(2*colors - 0.5, 0, 1), linewidth=0.5)
        ax.view_init(elevation, azimuth) #Elevation and Azimuth  
        j=j+1
    plt.axis('off')
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    plt.savefig(results_directory+save_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()


  
def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]



def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def save_image(image, image_path):
    return scipy.misc.imsave(image_path, image[0, :, :, 0])


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def imsave(images, size, path):
    return mpimg.imsave(path, merge(images, size))


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])


def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)


def inverse_transform(images):
    return np.clip(images, 0, 1)


def plot_losses(log_directory, results_directory, last_epoch=250):
	x = np.arange(0, last_epoch, 10) 

	mpl.style.use('fivethirtyeight')  # choose a preferred style
	mpl.rcParams['figure.dpi'] = 100
	# mpl.rcParams['savefig.dpi'] = 50

	mpl.rcParams['font.size'] = 10  # customise font size of a particular graph title, x-axis ticker and y-axis ticker
	mpl.rcParams['legend.fontsize'] = 10 # customise legend size
	mpl.rcParams['figure.titlesize'] = 15 # customise the size of suptitle

	#lines
	mpl.rcParams['lines.markersize'] = 10
	mpl.rcParams['legend.markerscale'] = 0.5
	mpl.rcParams['lines.markeredgewidth']  : 4
	fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[8,4],frameon = False)  # setup a single figure and define an empty axes
	ax.get_title(loc = "center")

	# customise each component manually
	ax.set_xlabel('Epoch') # give a label name to the x axis
	ax.set_ylabel('Loss') # give a label name to the x axis
	ax.legend(bbox_to_anchor=(0.99, 0.6)) #customise the legend location

	data = np.load(log_directory + 'total_loss.npy')
	idata = np.load(log_directory + 'image_loss.npy')
	ldata = np.load(log_directory + 'latent_loss.npy')

	line, = ax.plot(x, data,  label='Total Loss')
	line, = ax.plot(x, idata,  label='Reconstruction Loss')
	ax.legend()
	ax.patch.set_facecolor('w')
	ax.grid(b=None)


	line, = ax2.semilogy(x, ldata,  label='KL Loss')
	ax2.legend()
	ax2.set_xlabel('Epoch') # give a label name to the x axis
	ax2.patch.set_facecolor('w')
	ax2.grid(b=None)
	plt.tight_layout()  # avoid overlapping ticklabels, axis labels, and titles (can not control suptitle)

	#plt.show()
	fig.savefig(results_directory + 'losses.png', facecolor='w', edgecolor='none')



def save_interpolations(voxels_to_save, image,image2, decoder_outputs, binvox_size, results_directory, i, elevation=30, azimuth=45, mode='train', threshold=0.5):

    image = 255.*image
    image2 = 255.*image2

    save_file='{}.png'.format('Interpolation' + str(i)+'_'+mode+"_"+str(threshold))
    r, g, b = np.indices((binvox_size, binvox_size, binvox_size)) / binvox_size
    voxels = decoder_outputs>= threshold
    voxels = voxels.astype(np.bool)

    # Set up a figure 1/10 as tall as it is wide
    fig = plt.figure(figsize=plt.figaspect(1./10), frameon=False)


    ax = fig.add_subplot(1, 10, 1)
    # make the panes transparent
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)

    ax = fig.add_subplot(1, 10, 10)
    # make the panes transparent
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(image2, cmap='gray', vmin=0, vmax=255)


    for j in range(2,10):
        ax = fig.add_subplot(1, 10, j, projection='3d')
        nv = voxels_to_save[j-2]
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
        ax.axis('off')
        ax.set_xlim3d([0,binvox_size-1])
        ax.set_ylim3d([0,binvox_size-1])
        ax.set_zlim3d([0,binvox_size-1])

        colors = np.zeros(voxels[nv].reshape(-1,binvox_size,binvox_size).shape + (3,))
        colors[..., 0] = r
        colors[..., 1] = g
        colors[..., 2] = b
        ax.voxels(np.transpose(voxels[nv].reshape(-1,binvox_size,binvox_size),(2,0,1)), facecolors=colors, edgecolors=np.clip(2*colors - 0.5, 0, 1), linewidth=0.5)
        ax.view_init(elevation, azimuth) #Elevation and Azimuth  
        fig.add_axes(ax)
        j=j+1
    plt.axis('off')
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    plt.savefig(results_directory+save_file, bbox_inches='tight', pad_inches=0)
    plt.clf()




def save_interpolations2(voxels_to_save, image,image2, decoder_outputs, binvox_size, results_directory, i, elevation=30, azimuth=45, mode='train', threshold=0.5):

    image = 255.*image
    image2 = 255.*image2

    save_file='{}.png'.format('Interpolation' + str(i)+'_'+mode+"_"+str(threshold))
    r, g, b = np.indices((binvox_size, binvox_size, binvox_size)) / binvox_size
    voxels = decoder_outputs>= threshold
    voxels = voxels.astype(np.bool)

    # Set up a figure 1/6              as tall as it is wide
    fig = plt.figure(figsize=plt.figaspect(1./6), frameon=False)


    ax = fig.add_subplot(1, 6, 1)
    # make the panes transparent
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)

    ax = fig.add_subplot(1, 6, 6)
    # make the panes transparent
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(image2, cmap='gray', vmin=0, vmax=255)


    for j in range(2,6):
        ax = fig.add_subplot(1, 6, j, projection='3d')
        nv = voxels_to_save[j-2]
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,1.0)
        ax.axis('off')
        ax.set_xlim3d([0,binvox_size-1])
        ax.set_ylim3d([0,binvox_size-1])
        ax.set_zlim3d([0,binvox_size-1])

        colors = np.zeros(voxels[nv].reshape(-1,binvox_size,binvox_size).shape + (3,))
        colors[..., 0] = r
        colors[..., 1] = g
        colors[..., 2] = b
        ax.voxels(np.transpose(voxels[nv].reshape(-1,binvox_size,binvox_size),(2,0,1)), facecolors=colors, edgecolors=np.clip(2*colors - 0.5, 0, 1), linewidth=0.5)
        ax.view_init(elevation, azimuth) #Elevation and Azimuth  
        fig.add_axes(ax)
        j=j+1
    plt.axis('off')
    fig.subplots_adjust(bottom = 0)
    fig.subplots_adjust(top = 1)
    fig.subplots_adjust(right = 1)
    fig.subplots_adjust(left = 0)
    plt.savefig(results_directory+save_file, bbox_inches='tight', pad_inches=0)
    plt.clf()

