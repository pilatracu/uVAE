
##########################################################
"""
Author: Talip Ucar
Date: November 1, 2019
Description: 3D-VAE model is developed to learn 3D shapes from single images. 
Disclaimer: Some of the utility functions used in uvae_ops are taken from PrGAN github page: https://github.com/matheusgadelha/PrGAN 
"""
##########################################################
import tensorflow as tf
import numpy as np
import uvae_ops as ops
import tfrecords as tr
import glob
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
##########################################################
#Provide results and data directory. 
#Results will be saved under results directory
#Data directory is where the data set folder is located
#Place your datasets under img_dir_base 

main_dir = './results/'
img_dir_base = './data/'

##########################################################
##########################################################
# A simple command to run this code:
# python -u uvae.py -d all_datasets --train -e 15 -ims 28 -bs 64  -texture True -model_name 'uvae' > log_uvae
# where you can give any name for '-d', '-model_name' arguments as well as for log file 'log_3D_VAE'. 
# MNIST, CelebA, and Model40 datasets are hard coded, which will be updated in the future to provide them as option on the command line. 
##########################################################



parser = argparse.ArgumentParser(description='This program trains a uVAE model.')
parser.add_argument("-e", "--epochs", type=int, help="Number training epochs.", default=50)
parser.add_argument("-ims", "--image_size", type=int, help="Image size (single dimension).", default=137)
parser.add_argument("-bs", "--batch_size", type=int, help="Minibatch size.", default=64)
parser.add_argument("-bern", "--bernoulli", type=bool, help="To use Bernoulli dist at the output of decoder.", default=False)
parser.add_argument("-texture", "--texture", type=bool, help="To use Texture at the output of decoder.", default=False)
add_argument = parser.add_argument("-d", "--dataset", type=str,help="Dataset name.", default="chairs_canonical")
add_argument = parser.add_argument("-model_name", "--model_name", type=str, help="Model name", default="test")
add_argument = parser.add_argument("-img_dir", "--img_dir", type=str, help="Image directory", default="test_28x28x1")
parser.add_argument("--train", dest='train', action='store_true')
parser.add_argument("--sample", dest='sample', action='store_true')
parser.add_argument("--test", dest='test', action='store_true')
parser.add_argument("--transfer", dest='transfer', action='store_true')
parser.add_argument("--interpolate", dest='interpolate', action='store_true')
parser.add_argument("--rotation_interpolate", dest='rotation_interpolate', action='store_true')
parser.add_argument("--unit_interpolate", dest='unit_interpolate', action='store_true')
parser.set_defaults(train=False)




class uVAE:

    def __init__(self, sess=tf.Session(), image_size=(28, 28), z_size=20, n_iterations=50, dataset="None", batch_size=64, lrate=0.001, d_size=128, bernoulli = False, texture=False, model_name='test', img_dir='test_28x28x1', args=None):

        self.model_name = model_name    

        #Read in MNIST dataset
        self.img_dir = img_dir_base+'mnist/'
        self.database = input_data.read_data_sets(self.img_dir, one_hot = True)

        #Read in MNIST Fashion dataset
        self.img_dir2 = img_dir_base + 'MNIST_Fashion/'   
        self.database2 = input_data.read_data_sets(self.img_dir2, one_hot = True) 

        #Read in ModelNet40 dataset
        self.img_dir3 = img_dir_base + 'celeba_data_28x28x1/'
        self.img_dir4 = img_dir_base + 'chair_28x28x1/'
        self.img_dir5 = img_dir_base + 'car_28x28x1/'
        self.img_dir6 = img_dir_base + 'cup_28x28x1/'
        self.img_dir7 = img_dir_base + 'bowl_28x28x1/'
        self.img_dir8 = img_dir_base + 'person_28x28x1/'
        self.img_dir9 = img_dir_base + 'airplane_28x28x1/'
        self.img_dir_list = [self.img_dir3, self.img_dir4, self.img_dir5, self.img_dir6, self.img_dir7, self.img_dir8, self.img_dir9]
          
        #If true, include texture layer as final layer.
        self.texture = texture    

        if self.texture:
            self.model_name = self.model_name + '_texture'
        self.create_folders(main_dir,self.model_name)

      
        # Some variables are hard coded. The code will be updated later.
        self.bernoulli = bernoulli             
        self.lr=lrate
        self.alpha=1.0 
        self.beta=1.0
        self.image_size = image_size
        self.binvox_size = image_size[0]
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.session = sess
        self.base_dim = 512
        self.d_size = d_size
        self.z_size = z_size
        self.tau = 0.5
        self.dataset_name = dataset
        self.size = image_size[0]
        self.logpath = self.log_directory

        self.dataset_files = []
        for image_dir in self.img_dir_list:
            self.dataset_files = np.concatenate((self.dataset_files, glob.glob(image_dir +'/*.png')), axis=0)
        
        self.dataset_files = np.array(self.dataset_files)
        self.n_files = self.dataset_files.shape[0]  # 97080
        self.n_batches = self.n_files // self.batch_size


        # Lists and dictionares to collect data
        self.history = {}
        self.history["total_loss"] = []
        self.history["image_loss"] = []
        self.history["latent_loss"] = []
        self.history["stan_latent_loss"] = []


        self.num_channels = 1
        self.train_flag = tf.placeholder(tf.bool)

        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, self.size, self.size, self.num_channels], name='X')
        self.Yimg    = tf.placeholder(dtype=tf.float32, shape=[None, self.size, self.size, self.num_channels], name='Yimg')

        self.prior_mean = tf.zeros(shape=(self.batch_size, 3), dtype=tf.float32)
        self.prior_log_scale = tf.zeros(shape=(self.batch_size, 3), dtype=tf.float32)
        self.posterior_mean = tf.get_variable(name="posterior_mean", shape=(self.batch_size, 12), initializer=tf.zeros_initializer(), trainable=True, dtype=tf.float32)  
        self.posterior_log_scale = tf.get_variable(name="posterior_log_scale", shape=(self.batch_size, 12), initializer=tf.zeros_initializer(), trainable=True, dtype=tf.float32)


        self.Yimg_flat = tf.reshape(self.Yimg, shape=[-1, self.size * self.size * self.num_channels])
          

        self.sampled, self.mn, self.sd, self.z_theta = self.encoder(self.X_in, self.train_flag)
        if self.texture:
            self.final_imgs, self.voxels, self.textured_imgs = self.decoder(self.sampled, self.train_flag)
        else:
            self.final_imgs, self.voxels = self.decoder(self.sampled, self.train_flag)


        # Image Loss
        unreshaped_img = tf.reshape(self.final_imgs, [-1, self.size*self.size*self.num_channels])
        self.image_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(unreshaped_img, self.Yimg_flat), 1))

        # KL Loss
        self.latent_loss1 = tf.abs(tf.reduce_mean(tf.reduce_sum(self.mn, 1)))
        self.latent_loss2 = tf.abs(tf.reduce_mean( tf.reduce_sum(self.sd, 1))) 
        self.latent_loss = self.latent_loss1 + self.latent_loss2
        self.stan_latent_loss =  tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + self.sd - tf.square(self.mn) - tf.exp(self.sd), 1))


        self.latent_loss_theta1 = tf.abs(tf.reduce_mean(tf.reduce_sum(self.posterior_mean, 1)))
        self.latent_loss_theta2 = tf.abs(tf.reduce_mean( tf.reduce_sum(self.posterior_log_scale, 1))) 
        self.kl_loss_theta = self.latent_loss_theta1 + self.latent_loss_theta2


        if self.texture:
            unreshaped_timg = tf.reshape(self.textured_imgs, [-1, self.size*self.size*self.num_channels])
            self.image_loss2 = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(unreshaped_timg, self.Yimg_flat), 1))
            # Total Loss
            self.loss = self.image_loss + self.image_loss2 + self.beta*self.latent_loss + self.kl_loss_theta
        else:        
            self.loss = self.image_loss +  self.beta*self.latent_loss + self.kl_loss_theta


        # Optimizer used for VAE
        self.optimizer = tf.train.AdamOptimizer(self.lr)      
        grad_vars = self.optimizer.compute_gradients(self.loss)
        grad_vars = [
        (tf.clip_by_norm(grad, 1.0), var)
        if grad is not None else (grad, var)
        for grad, var in grad_vars]
        self.train_op = self.optimizer.apply_gradients(grad_vars)


        # Variational variable optimizer used to learn Theta
        self.variational_vars_optimizer = tf.train.AdamOptimizer(self.lr)
        self.variational_vars = [self.posterior_mean, self.posterior_log_scale] # list of variational variables
        self.variational_vars_update_op = self.variational_vars_optimizer.minimize(self.loss, var_list=self.variational_vars)

        # Collect data
        self.summary_posterior_std = tf.summary.histogram("post_std", self.posterior_log_scale)
        self.summary_posterior_mean = tf.summary.histogram("post_mean", self.posterior_mean)
        self.summary_z_theta = tf.summary.histogram("z_theta", self.z_theta)
        self.summary_mean = tf.summary.histogram("z_mean", self.mn)
        self.summary_std = tf.summary.histogram("z_std", self.sd)
        self.summary_sample = tf.summary.histogram("z_sample", self.sampled)
        self.summary_loss = tf.summary.scalar("loss", self.loss)
        self.summary_latent_loss1 = tf.summary.scalar("latent_loss1", self.latent_loss1)
        self.summary_latent_loss2 = tf.summary.scalar("latent_loss2", self.latent_loss2)
        self.summary_latent_loss = tf.summary.scalar("latent_loss", self.latent_loss)
        self.summary_stan_latent_loss = tf.summary.scalar("stan_latent_loss", self.stan_latent_loss)
        self.summary_img_loss = tf.summary.scalar("image_loss", self.image_loss)
        self.summary_all = tf.summary.merge_all()
        self.summ_writer = tf.summary.FileWriter(self.tensorboard_directory, self.session.graph)   

      
        # Save Model
        self.saver = tf.train.Saver(max_to_keep=25, write_version=tf.train.SaverDef.V2)

     


    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.model_directory)
        if ckpt and ckpt.model_checkpoint_path:
            print("Loading previous model...")
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print("Done.")
        else:
            print("No saved model found.")
        



    def create_folders(self, main_dir,model_name):

        # Model directory
        self.model_directory = main_dir+'checkpoint/'+model_name + '/'
        if not os.path.isdir(self.model_directory):
          os.mkdir(self.model_directory)

        # Results directory for saving images and voxels
        self.results_directory = main_dir+'results/'+model_name + '/'
        if not os.path.isdir(self.results_directory):
          os.mkdir(self.results_directory)
  

        # Results directory for saving images and voxels
        self.tensorboard_directory = main_dir+'results/'+model_name + '/' + 'summary/'
        if not os.path.isdir(self.results_directory):
          os.mkdir(self.results_directory)


        # Log directory for tensorboard: Remove previous one and create a new one.
        self.log_directory = main_dir+'log_'+model_name + '/'
        if not os.path.isdir(self.log_directory):
          os.mkdir(self.log_directory)


        # Interpolation results directory for saving images and voxels
        self.interpolation_directory = main_dir+'results/'+model_name + '/interpolations/'
        if not os.path.isdir(self.interpolation_directory):
          os.mkdir(self.interpolation_directory)


        # Interpolation results directory for saving images and voxels
        self.rotation_interpolation_directory = main_dir+'results/'+model_name + '/rotation_interpolations/'
        if not os.path.isdir(self.rotation_interpolation_directory):
          os.mkdir(self.rotation_interpolation_directory)

        # Interpolation results directory for saving images and voxels
        self.unit_interpolation_directory = main_dir+'results/'+model_name + '/unit_interpolations/'
        if not os.path.isdir(self.unit_interpolation_directory):
          os.mkdir(self.unit_interpolation_directory)

                
        
    def train(self):


        training_step = 0

        self.session.run(tf.global_variables_initializer())
        #self.load()
        training_losses = []
        image_losses = []
        latent_losses = []
        stan_latent_losses = []
        train_loss_per_epoch = []
        

        estimated_means = np.zeros((self.batch_size, self.z_size-1))
        estimated_std = np.zeros((self.batch_size, self.z_size-1))

        jj=0


                    #idxs_i = rand_idxs[batch_i * int(self.batch_size/2): (batch_i + 1) * int(self.batch_size/2)]
                    #sample_batch3 = ops.load_imgbatch(self.dataset_files[idxs_i], color=False)
                    #sample_batch3 = np.array(sample_batch3)
                    #sample_batch3 = sample_batch3.reshape(-1, 28*28)

                    #sample_batch, _ = self.database.train.next_batch(int(self.batch_size/4))
                    #sample_batch2, _ = self.database2.train.next_batch(int(self.batch_size/4))

                    #sample_batch = np.concatenate((sample_batch, sample_batch2), axis=0)
                    #sample_batch = np.concatenate((sample_batch, sample_batch3), axis=0)


        # Get a TEST SET to monitor the progress during training
        # Half of the batch should come from ModelNet40 datasets.
        # Random shiffle it before getting the batch.
        batch_i=0
        rand_idxs = np.random.permutation(range(self.n_files))
        idxs_i = rand_idxs[batch_i * int(self.batch_size/2): (batch_i + 1) * int(self.batch_size/2)]
        test_img4 = ops.load_imgbatch(self.dataset_files[idxs_i], color=False)
        test_img4 = np.array(test_img4)
        test_img4 = test_img4.reshape(-1, 28*28)


        # 1/4th of batch is from MNIST dataset
        test_img1, _ = self.database.train.next_batch(int(self.batch_size/4))
        # 1/4th of batch is from MNIST Fashion dataset
        test_img2, _ = self.database2.train.next_batch(int(self.batch_size/4))

        # Merge them all to have a complete batch consisting of all datasets.
        test_img3    = np.concatenate((test_img1, test_img2), axis=0)
        test_img3    = np.concatenate((test_img3, test_img4), axis=0)
        test_img3    = test_img3.reshape(-1, 28, 28, 1)


        for epoch1 in range(1):
            for epoch in range(self.n_iterations):

                rand_idxs = np.random.permutation(range(self.n_files))
                for batch_i in range(self.n_batches):
                    epoch=training_step


                    #Get TRAINING batch, half of which is from ModelNey40, and second half from MNIST and MNIST Fashion

                    #Get MNIST and MNIST Fashion
                    sample_batch, _ = self.database.train.next_batch(int(self.batch_size/4))
                    sample_batch2, _ = self.database2.train.next_batch(int(self.batch_size/4))

                    #Get ModelNet40
                    idxs_i = rand_idxs[batch_i * int(self.batch_size/2): (batch_i + 1) * int(self.batch_size/2)]
                    sample_batch3 = ops.load_imgbatch(self.dataset_files[idxs_i], color=False)
                    sample_batch3 = np.array(sample_batch3)
                    sample_batch3 = sample_batch3.reshape(-1, 28*28)

                    #Combine all batchs to have a complete batch
                    sample_batch = np.concatenate((sample_batch, sample_batch2), axis=0)
                    sample_batch = np.concatenate((sample_batch, sample_batch3), axis=0)

                    #Randomly shuffle the batch and reshape
                    np.random.shuffle(sample_batch)
                    sample_batch = sample_batch.reshape(-1, 28, 28, 1)
                    
                    #Update the parameters of network used to learn Theta. We can update multiple times, but one update seems OK.
                    for _ in range(1):
                        self.session.run(self.variational_vars_update_op,  feed_dict = {self.X_in: sample_batch, self.Yimg: sample_batch, self.train_flag: True})  

                    #Update the parameters of VAE used to learn shape
                    self.session.run(self.train_op,  feed_dict = {self.X_in: sample_batch, self.Yimg: sample_batch, self.train_flag: True})    
                    summ, training_loss, images_hat, train_image_loss, train_latent_loss,train_stan_latent_loss, mean_loss, std_loss = self.session.run([self.summary_all, self.loss, self.final_imgs, self.image_loss, self.latent_loss, self.stan_latent_loss, self.latent_loss1, self.latent_loss2], feed_dict = {self.X_in: sample_batch,  self.Yimg: sample_batch, self.train_flag: False})
                    

                    # Collect loss values and statistics
                    training_losses.append(training_loss)
                    image_losses.append(train_image_loss)
                    latent_losses.append(train_latent_loss)
                    stan_latent_losses.append(train_stan_latent_loss)
                    
                    computed_means = self.mn.eval(session=self.session, feed_dict = {self.X_in: sample_batch,  self.Yimg: sample_batch, self.train_flag: False})
                    computed_std = self.sd.eval(session=self.session, feed_dict   = {self.X_in: sample_batch,  self.Yimg: sample_batch, self.train_flag: False})

                    estimated_means = (jj*estimated_means + computed_means)/(jj+1)  
                    estimated_std =   (jj*estimated_std + computed_std)/(jj+1)  
                    jj=jj+1					   


                    if epoch % 50 ==0:
                    	test_batch = test_img3
                    	summ= self.session.run(self.summary_all, feed_dict = {self.X_in: sample_batch,  self.Yimg: sample_batch, self.train_flag: False})
                    	self.summ_writer.add_summary(summ, epoch)

                    	post_means = self.posterior_mean.eval(session=self.session)
                    	post_std = self.posterior_log_scale.eval(session=self.session)

                    	rendered_images = self.final_imgs.eval(session=self.session, feed_dict = {self.X_in: test_batch,  self.Yimg: test_batch, self.train_flag: False})
                    	rendered_images = np.array(rendered_images)
                    	rendered_images =  rendered_images/np.amax(rendered_images)


                    	voxels = self.voxels.eval(session=self.session, feed_dict = {self.X_in: test_batch,  self.Yimg: test_batch, self.train_flag: False})
                    	voxels = np.array(voxels)
                    	ops.save_images(rendered_images,  [8, 8], self.results_directory+"/output_projection_{}.png".format(epoch))
                    	ops.save_images(test_batch, [8, 8], self.results_directory + "input_data_{}.png".format(epoch))

                    	if self.texture:
                    		rendered_images2 = self.textured_imgs.eval(session=self.session, feed_dict = {self.X_in: test_batch,  self.Yimg: test_batch, self.train_flag: False})
                    		rendered_images2 = np.array(rendered_images2)
                    		ops.save_images(rendered_images2,  [8, 8], self.results_directory+"/output_texturizer_{}.png".format(epoch))
                    		rendered_images2 = rendered_images2/np.amax(rendered_images2)
                    		rendered_images=rendered_images2



                    	texture_image1=rendered_images[3].reshape(self.size, self.size)
                    	texture_image14=rendered_images[4].reshape(self.size, self.size)
                    	texture_image15=rendered_images[5].reshape(self.size, self.size)

                    	texture_image2=rendered_images[19].reshape(self.size, self.size)
                    	texture_image20=rendered_images[20].reshape(self.size, self.size)
                    	texture_image21=rendered_images[21].reshape(self.size, self.size)

                    	texture_image3=rendered_images[39].reshape(self.size, self.size)
                    	texture_image4=rendered_images[49].reshape(self.size, self.size)

                    	ops.save_image_voxel_grid(texture_image1, voxels[3], self.binvox_size, self.results_directory, 3, elevation=0, azimuth=0,    mode='rendered'+str(epoch), threshold=0.1)
                    	ops.save_image_voxel_grid(texture_image14, voxels[4], self.binvox_size, self.results_directory, 4, elevation=0, azimuth=0,   mode='rendered'+str(epoch), threshold=0.1)
                    	ops.save_image_voxel_grid(texture_image15, voxels[5], self.binvox_size, self.results_directory, 5, elevation=0, azimuth=0,   mode='rendered'+str(epoch), threshold=0.1)
                    	ops.save_image_voxel_grid(texture_image2, voxels[19], self.binvox_size, self.results_directory, 19, elevation=0, azimuth=0,  mode='rendered'+str(epoch), threshold=0.1)
                    	ops.save_image_voxel_grid(texture_image20, voxels[20], self.binvox_size, self.results_directory, 20, elevation=0, azimuth=0, mode='rendered'+str(epoch), threshold=0.1)
                    	ops.save_image_voxel_grid(texture_image21, voxels[21], self.binvox_size, self.results_directory, 21, elevation=0, azimuth=0, mode='rendered'+str(epoch), threshold=0.1)
                    	ops.save_image_voxel_grid(texture_image3, voxels[39], self.binvox_size, self.results_directory, 39, elevation=0, azimuth=0,  mode='rendered'+str(epoch), threshold=0.1)
                    	ops.save_image_voxel_grid(texture_image4, voxels[49], self.binvox_size, self.results_directory, 49, elevation=0, azimuth=0,  mode='rendered'+str(epoch), threshold=0.1)

                    	ops.save_voxels_as_image(voxels[3], self.binvox_size, self.results_directory, training_step, elevation=0, azimuth=0,  mode='train0', threshold=0.1)
                    	ops.save_voxels_as_image(voxels[19], self.binvox_size, self.results_directory, training_step, elevation=0, azimuth=0, mode='train1', threshold=0.101)
                    	ops.save_voxels_as_image(voxels[39], self.binvox_size, self.results_directory, training_step, elevation=0, azimuth=0, mode='train2', threshold=0.101)
                    	ops.save_voxels_as_image(voxels[49], self.binvox_size, self.results_directory, training_step, elevation=0, azimuth=0, mode='train3', threshold=0.102)


                    	if epoch % 100 ==0:
                    	    print(estimated_means)
                    	    print(computed_std)
                    	    print("Saving checkpoint...")
                    	    print(mean_loss, std_loss)
                    	    np.save(os.path.join(self.logpath, "estimated_means.npy"), np.array(estimated_means))
                    	    np.save(os.path.join(self.logpath, "estimated_std.npy"), np.array(estimated_std))
                    	    np.save(os.path.join(self.logpath, "post_means.npy"), np.array(post_means))
                    	    np.save(os.path.join(self.logpath, "post_std.npy"), np.array(post_std))

                    	    self.saver.save(self.session, self.model_directory + 'model.ckpt', global_step=training_step)
                    	    print("------------Model is saved-------------")

                    if epoch % 50 ==0:
                    	total_mean = np.mean(training_losses)
                    	image_mean = np.mean(image_losses)
                    	latent_mean = np.mean(latent_losses)
                    	stan_latent_mean = np.mean(stan_latent_losses)


                    	self.history["total_loss"].append(total_mean)
                    	self.history["image_loss"].append(image_mean)
                    	self.history["latent_loss"].append(latent_mean)
                    	self.history["stan_latent_loss"].append(stan_latent_mean)

                    	print("EPOCH[{}],".format(epoch))
                    	print("Training loss mean: {}".format(total_mean))
                    	print("Image loss mean: {}".format(image_mean))
                    	print("Latent loss mean: {}".format(latent_mean))
                    	print("Standard Latent loss mean: {}".format(stan_latent_mean))

                    	np.save(os.path.join(self.logpath, "total_loss.npy"), np.array(self.history["total_loss"]))
                    	np.save(os.path.join(self.logpath, "image_loss.npy"), np.array(self.history["image_loss"]))
                    	np.save(os.path.join(self.logpath, "latent_loss.npy"), np.array(self.history["latent_loss"]))
                    	np.save(os.path.join(self.logpath, "stan_latent_loss.npy"), np.array(self.history["stan_latent_loss"]))
                    	np.save(os.path.join(self.logpath, "image_loss_per_batch.npy"), np.array(image_losses))
                    	np.save(os.path.join(self.logpath, "latent_loss_per_batch.npy"), np.array(latent_losses))
                    	np.save(os.path.join(self.logpath, "stan_latent_loss_per_batch.npy"), np.array(stan_latent_losses))


                    training_step += 1



        
    def encoder(self, image, train, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            
            reshaped_img = tf.reshape(image, [self.batch_size, self.image_size[0], self.image_size[1], 1])
            h0 = ops.conv2d(reshaped_img, self.d_size, name='e_h0_conv')
            h0 = ops.lrelu(tf.contrib.layers.batch_norm(h0,decay=0.9, epsilon=1e-5, is_training=train,center= True, scale=True))  
            h1 = ops.conv2d(h0, self.d_size, name='e_h1_conv')
            h1 = ops.lrelu(tf.contrib.layers.batch_norm(h1,decay=0.9, epsilon=1e-5, is_training=train,center= True, scale=True))
            h2 = ops.conv2d(h1, self.d_size, name='e_h2_conv')
            h2 = ops.lrelu(tf.contrib.layers.batch_norm(h2,decay=0.9, epsilon=1e-5, is_training=train,center= True, scale=True))
            h3 = ops.conv2d(h2, self.d_size, name='e_h3_conv')
            h3_tensor = ops.lrelu(tf.contrib.layers.batch_norm(h3,decay=0.9, epsilon=1e-5, is_training=train,center= True, scale=True))

            x = tf.contrib.layers.flatten(h3_tensor)
            mn = tf.layers.dense(x, units=self.z_size-1)
            mn = tf.clip_by_norm(mn, 6.0*tf.sqrt(1.0*self.z_size), axes=1)

            sd = tf.layers.dense(x, units=self.z_size-1)   
            sd = tf.clip_by_norm(sd, 1.0*tf.sqrt(1.0*self.z_size), axes=1)

            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], self.z_size-1]), stddev=1.0) 
            z  = mn + tf.multiply(epsilon, tf.exp(0.5*sd))

            # Implement neural network used for Theta here as well since Theta can be implemented as a 
            # function of data ,i.e. using encoder output, in the future.
            # z_theta code below needs to be pulled out of encoder() in the future since it is independent of encoder()

            epsilon_theta = tf.random_normal(tf.stack([tf.shape(x)[0], 12]), stddev=1.0) 
            z_theta  = self.posterior_mean + tf.multiply(epsilon_theta, tf.exp(0.5*self.posterior_log_scale))
            z_theta  = tf.layers.dense(z_theta, units=10, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            z_theta  = tf.layers.dense(z_theta, units=1, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            z = tf.concat([z, z_theta], 1)
           
            return z, mn, sd, z_theta


    def decoder(self, z_enc, train, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            base_filters = self.d_size
            h0 = ops.linear(z_enc[:, 0:(self.z_size-1)], self.z_size-1, 7*7*7*base_filters, scope='d_f0')
            h0 = tf.reshape(h0, [self.batch_size, 7, 7, 7, base_filters])
            h0 = ops.lrelu(tf.contrib.layers.batch_norm(h0,decay=0.9, epsilon=1e-5, is_training=train,center= True, scale=True))    
            h1 = ops.deconv3d(h0, [self.batch_size, 14, 14, 14, int(base_filters/2)], name='d_h1')
            h1 = ops.lrelu(tf.contrib.layers.batch_norm(h1,decay=0.9, epsilon=1e-5, is_training=train,center= True, scale=True))
            h4 = ops.deconv3d(h1, [self.batch_size, 28, 28, 28, 1], name='d_h4')
            h4 = tf.nn.sigmoid(h4) * (1.0/self.tau)
            self.voxels = tf.reshape(h4, [self.batch_size, 28, 28, 28])
            v = z_enc[:, self.z_size-1]

            rendered_imgs = []
            for i in range(self.batch_size):
                img = ops.project(ops.transform_volume(self.voxels[i], ops.rot_matrix(v[i])),
                        self.tau)
                rendered_imgs.append(img)
            # Scale to image size  using linear layer  
            self.final_imgs = tf.reshape(rendered_imgs, [self.batch_size, self.size, self.size, 1])

            if self.texture:
                silhoutte = tf.reshape(self.final_imgs, [-1, 28*28])
                silhoutte = tf.layers.dense(inputs=silhoutte, units=1000, activation=tf.nn.leaky_relu, name="dec_nn1", reuse=reuse)
                textured_images = tf.layers.dense(inputs=silhoutte, units=28*28, activation=tf.nn.sigmoid, name="dec_nn2", reuse=reuse)
                textured_images  = tf.reshape(textured_images , [-1, 28, 28, 1])

                return self.final_imgs, self.voxels, textured_images
            else:
                return self.final_imgs, self.voxels



    def tsne(self):
        self.session.run(tf.initialize_all_variables())
        self.load()

        n_components=2

        for ij in range(3000):
            x_sample, y_sample1 = database.test.next_batch(int(self.batch_size/2))
            x_sample2, y_sample1 = database.test.next_batch(int(self.batch_size/2))
            x_sample = np.concatenate((x_sample, x_sample2), 0)

            test_batch = x_sample

            y_sample = np.ones((1,self.batch_size))
            y_sample[self.batch_size/2:]=2

            interim = self.sampled.eval(session=self.session, feed_dict = {self.X_in: test_batch,  self.Yimg: test_batch, self.train_flag: False})
  
            if ij==0:
                samples_array = interim
                colors = y_sample

            else: 
                samples_array = np.vstack((samples_array,interim))
                colors = np.hstack((colors,y_sample))
    

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6,6],frameon = False)  # setup a single figure and define an empty axes
        mds = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
        Y = mds.fit_transform(samples_array[0:8000])
        ax = fig.add_subplot(258)
        plt.scatter(Y[:, 0], Y[:, 1], c=colors[0:8000], cmap=plt.cm.Spectral)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        fig.savefig(self.results_directory+'clustering.png', facecolor='w', edgecolor='none')

       



    def sample(self):
        self.session.run(tf.initialize_all_variables())
        self.load()
        epoch=1

        sample_batch, _ = self.database2.train.next_batch(int(self.batch_size/2))
        sample_batch2, _ = self.database2.train.next_batch(int(self.batch_size/2))

        sample_batch = np.concatenate((sample_batch, sample_batch2), axis=0)
        np.random.shuffle(sample_batch)
        sample_batch = sample_batch.reshape(-1, 28, 28, 1)
        test_batch = sample_batch

        sampled2 = self.sampled.eval(session=self.session, feed_dict = {self.X_in: test_batch,  self.Yimg: test_batch, self.train_flag: False})



        estimated_means = np.load(os.path.join(self.log_directory,'estimated_means.npy'))
        estimated_std = np.load(os.path.join(self.log_directory,'estimated_std.npy'))

        voxels_to_save = range(0, 64)
        zp = np.random.normal(0, 4.0, [self.batch_size, self.z_size-1])
        zp2 = np.random.normal(0, 5.0, [self.batch_size, self.z_size-1]) 
        zp3 = np.random.normal(0, 7.0, [self.batch_size, self.z_size-1]) 
        zp4 = np.random.normal(0, 3.0, [self.batch_size, self.z_size-1]) 

        zp_theta = np.zeros((self.batch_size, 1)) #np.random.uniform(-1,1,(self.batch_size,1))


        new_samples = np.concatenate((zp, zp_theta), 1)
        new_samples2 = np.concatenate((zp2, zp_theta), 1)
        new_samples3 = np.concatenate((zp3, zp_theta), 1)
        new_samples4 = np.concatenate((zp4, zp_theta), 1)


        sampled2 = sampled2[19,:]
        sampled2 =  np.tile(sampled2, (self.batch_size, 1))   

        new_samples = sampled2 + new_samples
        new_samples2 = sampled2 + new_samples2
        new_samples3 = sampled2 + new_samples3
        new_samples4 = sampled2 #+ new_samples4



        rendered_images = self.final_imgs.eval(session=self.session, feed_dict = {self.sampled: new_samples, self.train_flag: False})
        rendered_images2 = self.final_imgs.eval(session=self.session, feed_dict = {self.sampled: new_samples2, self.train_flag: False})
        rendered_images3 = self.final_imgs.eval(session=self.session, feed_dict = {self.sampled: new_samples3, self.train_flag: False})
        rendered_images4 = self.final_imgs.eval(session=self.session, feed_dict = {self.sampled: new_samples4, self.train_flag: False})

        if self.texture:
            rendered_images = self.textured_imgs.eval(session=self.session, feed_dict = {self.sampled: new_samples, self.train_flag: False})
            rendered_images2 = self.textured_imgs.eval(session=self.session, feed_dict = {self.sampled: new_samples2, self.train_flag: False})
            rendered_images3 = self.textured_imgs.eval(session=self.session, feed_dict = {self.sampled: new_samples3, self.train_flag: False})
            rendered_images4 = self.textured_imgs.eval(session=self.session, feed_dict = {self.sampled: new_samples4, self.train_flag: False})


        sampled_voxels = self.voxels.eval(session=self.session, feed_dict = {self.sampled: new_samples, self.train_flag: False})
            		
        rendered_images = np.array(rendered_images)
        rendered_images2 = np.array(rendered_images2)
        rendered_images3 = np.array(rendered_images3)
        rendered_images4 = np.array(rendered_images4)

        ops.save_images(rendered_images, [8, 8], self.results_directory+"/{}_sampled_{}.png".format(self.dataset_name, epoch))
        ops.save_images(rendered_images2, [8, 8], self.results_directory+"/{}_sampled_{}.png".format(self.dataset_name, epoch+1))
        ops.save_images(rendered_images3, [8, 8], self.results_directory+"/{}_sampled_{}.png".format(self.dataset_name, epoch+2))
        ops.save_images(rendered_images4, [8, 8], self.results_directory+"/{}_sampled_{}.png".format(self.dataset_name, epoch+3))


        for v in voxels_to_save:
            sampled_image = rendered_images[v]
            sampled_image = sampled_image.reshape(self.size, self.size)
            ops.save_image_voxel_grid(sampled_image, sampled_voxels[v], self.binvox_size, self.results_directory, v, elevation=0, azimuth=0,  mode='sampled', threshold=0.1)
            ops.save_image_voxel_grid(sampled_image, sampled_voxels[v], self.binvox_size, self.results_directory, v, elevation=30, azimuth=0,  mode='sampled', threshold=0.101)
            ops.save_image_voxel_grid(sampled_image, sampled_voxels[v], self.binvox_size, self.results_directory, v, elevation=90, azimuth=0,  mode='sampled', threshold=0.102)



    def rotation_interpolate(self):
        self.session.run(tf.initialize_all_variables())
        self.load()

		
        epoch=10000000

        sample_batch, _ = self.database.train.next_batch(int(self.batch_size/2))
        sample_batch2, _ = self.database2.train.next_batch(int(self.batch_size/2))

        sample_batch = np.concatenate((sample_batch, sample_batch2), axis=0)
        np.random.shuffle(sample_batch)
        sample_batch = sample_batch.reshape(-1, 28, 28, 1)
        test_batch = sample_batch

        sampled2 = self.sampled.eval(session=self.session, feed_dict = {self.X_in: test_batch,  self.Yimg: test_batch, self.train_flag: False})



        epoch=1		
        voxels_to_save = [0, 9, 18, 27, 31, 36, 45, 54, 63]
        voxels_to_save2 = [0, 7, 15, 23, 31]

        theta = np.linspace(-1.,1.,self.batch_size)

        for k in range(self.batch_size):

                sampled = sampled2[k,0:-1]
                sampled = np.tile(sampled, (self.batch_size, 1))   
                sampled = np.concatenate((sampled, theta.reshape(self.batch_size,1)), axis=1) 

                rendered_images = self.final_imgs.eval(session=self.session, feed_dict = {self.sampled: sampled, self.train_flag: False})
                rendered_images = np.array(rendered_images)
                ops.save_images(rendered_images, [8, 8], self.rotation_interpolation_directory+"/{}_sampled_rotation_{}.png".format(self.dataset_name, k))
                images_to_interpolate=rendered_images

                if self.texture:
                    rendered_images2 = self.textured_imgs.eval(session=self.session, feed_dict = {self.sampled: sampled, self.train_flag: False})
                    rendered_images2 = np.array(rendered_images2)
                    ops.save_images(rendered_images2, [8, 8], self.rotation_interpolation_directory+"/{}_sampled_rotation_{}_textured.png".format(self.dataset_name, k))
                    images_to_interpolate=rendered_images2


                sampled_voxels = self.voxels.eval(session=self.session, feed_dict = {self.sampled: sampled, self.train_flag: False})
                    		
                sampled_image1 = images_to_interpolate[voxels_to_save2[0]]
                sampled_image2 = images_to_interpolate[voxels_to_save2[-1]]

                sampled_image1 = sampled_image1.reshape(self.size, self.size)
                sampled_image2 = sampled_image2.reshape(self.size, self.size)

                ops.save_interpolations2(voxels_to_save2, sampled_image1,sampled_image2, sampled_voxels, self.binvox_size, self.rotation_interpolation_directory, k, elevation=0, azimuth=0, mode='sampled', threshold=0.1)
                ops.save_interpolations2(voxels_to_save2, sampled_image1,sampled_image2, sampled_voxels, self.binvox_size, self.rotation_interpolation_directory, k, elevation=30, azimuth=0, mode='sampled', threshold=0.101)



    def unit_interpolate(self):
        self.session.run(tf.initialize_all_variables())
        self.load()
		
        epoch=1		
        voxels_to_save = [0, 9, 18, 27, 36, 45, 54, 63]
        results = np.linspace(-10.,10.,self.batch_size)
        estimated_means = np.load(os.path.join(self.log_directory,'estimated_means.npy'))
        estimated_post_means = np.load(os.path.join(self.log_directory,"post_means.npy"))
        estimated_post_std = np.load(os.path.join(self.log_directory,"post_std.npy"))


        for k in range(2,self.z_size-1):
 
                zp = np.random.normal(0, 4.0, [1, self.z_size-k]) # + estimated_means[1, 0:self.z_size-k]
                zp = np.tile(zp, (self.batch_size, 1))   

                zp2 = np.random.normal(0, 4.0, [1, k-2]) #+ estimated_means[1,self.z_size-k+1:]  
                zp2 = np.tile(zp2, (self.batch_size, 1)) 
                zp_theta =  np.zeros((self.batch_size, 1)) #np.random.uniform(-1,1,(self.batch_size,1))
    
                interpolations = np.array(results)
                interpolations = np.concatenate((zp, interpolations.reshape(self.batch_size,1)), axis=1) 
                interpolations = np.concatenate((interpolations, zp2), axis=1) 
                interpolations = np.concatenate((interpolations, zp_theta), axis=1) 



                rendered_images = self.final_imgs.eval(session=self.session, feed_dict = {self.sampled: interpolations, self.train_flag: False})
                rendered_images = np.array(rendered_images)
                ops.save_images(rendered_images, [8, 8], self.unit_interpolation_directory+"/{}_sampled_{}.png".format(self.dataset_name, k))
                images_to_interpolate=rendered_images

                if self.texture:
                    rendered_images2 = self.textured_imgs.eval(session=self.session, feed_dict = {self.sampled: interpolations, self.train_flag: False})
                    rendered_images2 = np.array(rendered_images2)
                    ops.save_images(rendered_images2, [8, 8], self.unit_interpolation_directory+"/{}_sampled_{}_textured.png".format(self.dataset_name, k))
                    images_to_interpolate=rendered_images2


                sampled_voxels = self.voxels.eval(session=self.session, feed_dict = {self.sampled: interpolations, self.train_flag: False})

                sampled_image1 = images_to_interpolate[voxels_to_save[0]]
                sampled_image2 = images_to_interpolate[voxels_to_save[-1]]

                sampled_image1 = sampled_image1.reshape(self.size, self.size)
                sampled_image2 = sampled_image2.reshape(self.size, self.size)


                ops.save_interpolations(voxels_to_save, sampled_image1,sampled_image2, sampled_voxels, self.binvox_size, self.unit_interpolation_directory, k, elevation=0, azimuth=0, mode='sampled', threshold=0.1)
                ops.save_interpolations(voxels_to_save, sampled_image1,sampled_image2, sampled_voxels, self.binvox_size, self.unit_interpolation_directory, k, elevation=30, azimuth=0, mode='sampled', threshold=0.101)



    def test(self):
        self.session.run(tf.initialize_all_variables())
        self.load()
		
        #Use a dummy epoch number for naming purposes. Will clean this up in the future.
        epoch=10000000

        sample_batch, _ = self.database.train.next_batch(int(self.batch_size/2))
        sample_batch2, _ = self.database2.train.next_batch(int(self.batch_size/2))

        sample_batch = np.concatenate((sample_batch, sample_batch2), axis=0)
        np.random.shuffle(sample_batch)
        sample_batch = sample_batch.reshape(-1, 28, 28, 1)

                
        test_batch = sample_batch
                	
        rendered_images = self.final_imgs.eval(session=self.session, feed_dict = {self.X_in: test_batch,  self.Yimg: test_batch, self.train_flag: False})
        rendered_images = np.array(rendered_images)
        rendered_images =  rendered_images/np.amax(rendered_images)


        voxels = self.voxels.eval(session=self.session, feed_dict = {self.X_in: test_batch,  self.Yimg: test_batch, self.train_flag: False})
        voxels = np.array(voxels)
        ops.save_images(rendered_images,  [8, 8], self.results_directory+"/test_renderImgs_projection_{}_{}.png".format(self.dataset_name, epoch))
        ops.save_images(sample_batch, [8, 8], self.results_directory + "test_input_data_{}.png".format(epoch))

        if self.texture:
            rendered_images2 = self.textured_imgs.eval(session=self.session, feed_dict = {self.X_in: test_batch,  self.Yimg: test_batch, self.train_flag: False})
            rendered_images2 = np.array(rendered_images2)
            ops.save_images(rendered_images2,  [8, 8], self.results_directory+"/test_renderedImgs_texturizer_{}_{}_textured.png".format(self.dataset_name, epoch))
            rendered_images2 = rendered_images2/np.amax(rendered_images2)
            rendered_images=rendered_images2


        
        for v in range(64):
            texture_image=rendered_images[v].reshape(self.size, self.size)
            ops.save_image_voxel_grid(texture_image, voxels[v], self.binvox_size, self.results_directory, v, elevation=30, azimuth=0,  mode=str(epoch+v)+'test_1_', threshold=0.1)
            ops.save_image_voxel_grid(texture_image, voxels[v], self.binvox_size, self.results_directory, v, elevation=0, azimuth=0,  mode=str(epoch+v)+'test_2_', threshold=0.1)




def main():
    args = parser.parse_args()
    rgan = uVAE(n_iterations=args.epochs, 
            batch_size=args.batch_size, 
            image_size=(args.image_size, args.image_size),
            dataset=args.dataset, texture=args.texture, model_name=args.model_name, img_dir=args.img_dir, args=args)

    if args.train:
        rgan.train()
    if args.test:
        rgan.test()
    if args.transfer:
        rgan.transfer()
    if args.sample:
        rgan.sample()
    if args.interpolate:
        rgan.interpolate()
    if args.rotation_interpolate:
        rgan.rotation_interpolate()
    if args.unit_interpolate:
        rgan.unit_interpolate()

if __name__ == '__main__':
    main()





