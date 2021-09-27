#!/usr/bin/python

# Used as a test for conditional variational autoencoders
# Author: Matt Leming
# Used code from https://github.com/nnormandin/Conditional_VAE
# This script should be able to run on its own. Be sure to put it in its own
# folder. Requires Pandas, Matplotlib, Numpy.
# 
# This script generates image data to be input into a conditional variational
# autoencoder. The data is images that may or may not have a green rectangle
# and may or may not have a red ellipse. This script generates these images,
# records the ten parameters of each image (presence of a rectangle/ellipse (2),
# x/y coordinates for each (4), width/height for each (4)), and 
#
# The general goal of this project is to test out whether changing the label
# of an input can change the output image as expected. In this toy example,
# The goal was to simply remove a green rectangle while preserving the red
# ellipse, if it was present.

import os,imageio,warnings,matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint,choice
from PIL import Image,ImageDraw
warnings.simplefilter(action='ignore', category=FutureWarning)
matplotlib.use('Agg')


if False:
	# These imports aren't totally tested
	import keras
	from keras.layers import *
	from keras.models import Model,load_model
	from keras import backend as K
else:
	import tensorflow.keras as keras
	from tensorflow.keras.layers import *
	from tensorflow.keras.models import Model,load_model
	from tensorflow.keras import backend as K
	import tensorflow as tf

# Number of latent dimensions. Will visualize the first two in a plot
n_z = 2
assert(n_z >= 2)
batch_size = 256
# Dimension of the input image. The neural network is designed for 32x32, so
# altering this will probably mean that the convolutional parameters later
# need to be altered.
dim = (32,32)
# check_for_save_and_load_model = "Check for, save, and load model"
# Pick up where you left off. Set to False if you want to start everything anew
# Note: broken -- ignore
check_for_save_and_load_model = False
plot_latent_space = True
verbose = True
# Trains a discriminator in a GAN sense to make outputs like the training set
# Not yet implemented -- ignore
train_discriminator = False
# Terminates when training gets below this loss threshold, or when max_epochs
# has been reached
loss_lim = 0.001
max_epochs = 1000
# "div" is the number of images in the test set, with the remainder going into
# the training set. "num_random" is just the total number of images generated
# Note that 1/4 of them will be the same blank white image.
num_random_images = 5000
div = 1000
optimizer = "nadam"
assert(div < num_random_images)

# Custom layer function for sampling in the variational autoencoder
def sample_z(args):
	mu,l_sigma = args
	eps = K.random_normal(shape=(K.shape(mu)[0], n_z), mean=0., stddev=1.)
	return mu + K.exp(l_sigma / 2) * eps

# Current versions of Keras require this to pass extra arguments into loss
# functions, unfortunately

def disc_loss(l_sigma,mu):
	def sub_disc_loss(y_true,y_pred):
		bcr = K.binary_crossentropy(y_true,y_pred)
		kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)
		return bcr + kl
	return sub_disc_loss

def vae_loss(l_sigma,mu):
	def sub_vae_loss(y_true, y_pred):
		recon = K.mean(K.exp(y_true - y_pred),axis=(1,2,3))
		kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)
		return recon + kl
	return sub_vae_loss

class VAE(Model):
	def __init__(self, encoder, decoder, **kwargs):
		super(VAE, self).__init__(**kwargs)
		self.encoder = encoder
		self.decoder = decoder
		self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
		self.reconstruction_loss_tracker = keras.metrics.Mean(
			name="reconstruction_loss"
		)
		self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

	@property
	def metrics(self):
		return [
			self.total_loss_tracker,
			self.reconstruction_loss_tracker,
			self.kl_loss_tracker,
		]

	def train_step(self, data):
		data1,conf = data
		data1,conf = data1
		with tf.GradientTape() as tape:
			z_mean, z_log_var, z = self.encoder([data1,conf])
			print("z_mean.shape: %s" % str(z_mean.shape))
			print("z_log_var.shape: %s" % str(z_log_var.shape))
			print("z.shape: %s" % str(z.shape))
			print("conf.shape: %s" % str(conf.shape))
			print("data1: %s" %str(data1))
			reconstruction = self.decoder([z,conf])
			reconstruction_loss = tf.reduce_mean(
				tf.reduce_sum(
					keras.losses.binary_crossentropy(data1, reconstruction), axis=(1, 2)
				)
			)
			kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
			kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
			total_loss = reconstruction_loss + kl_loss
		grads = tape.gradient(total_loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
		self.total_loss_tracker.update_state(total_loss)
		self.reconstruction_loss_tracker.update_state(reconstruction_loss)
		self.kl_loss_tracker.update_state(kl_loss)
		return {
			"loss": self.total_loss_tracker.result(),
			"reconstruction_loss": self.reconstruction_loss_tracker.result(),
			"kl_loss": self.kl_loss_tracker.result(),
		}
	def predict(self,data):
		data1,conf = data
		z_mean, z_log_var, z = self.encoder([data1,conf])
		return self.decoder.predict([z,conf])

# Folder this file is in
working_dir = os.path.dirname(os.path.realpath(__file__))

# The image data is stored in image folder; a composite of the image, predicted
# image, and predicted image with a deliberate alteration is stored in
# image_predictions
image_folder = os.path.join(working_dir,"images")
image_prediction_folder = os.path.join(working_dir,"image_predictions")
gif_folder = os.path.join(working_dir,"gifs")
model_folder = os.path.join(working_dir,"models")
for folder in [image_folder,image_prediction_folder,gif_folder,model_folder]:
	if not os.path.isdir(folder): os.makedirs(folder)

epoch_number_file = os.path.join(model_folder,'epoch.npy')
model_file = os.path.join(model_folder,'model.h5')
latent_space_plot = os.path.join(working_dir,'latent_space_plot.png')
pd_path = os.path.join(working_dir,"data_pd.pkl")

# Images that will be compiled into gifs as training goes. This just selects
# the first 20 images and the latent space plot
gif_list = [os.path.join(image_prediction_folder,
			"im_%.5d.png"%i) for i in range(20)]
if plot_latent_space: gif_list.append(latent_space_plot)

# This function constructs gifs out of particular images as they're saved
def add_to_gif(image_file,gif_file = None,new_gif=False):
	assert(os.path.isfile(image_file))
	if gif_file is None:
		gif_file = os.path.basename(os.path.splitext(image_file)[0] + ".gif")
		gif_file = os.path.join(gif_folder,gif_file)
	image = imageio.imread(image_file)
	images = []
	if os.path.isfile(gif_file) and not (new_gif):
		im = imageio.get_reader(gif_file)
		for frame in im:
			images.append(frame)
	images.append(image)
	imageio.mimsave(gif_file,images,fps = 3)

# This code generates data and stores it in image_folder. Data about the plots
# (e.g., presence of squares and ellipses, as well as their coordinates) are
# stored in a Pandas dataframe. If the dataframe is present, this is skipped.
# Delete the dataframe to generate new data
columns = ["rec_x","rec_y","rec_w","rec_h","has_rec","ell_x","ell_y",
	"ell_w","ell_h","has_ell"]

# The labels to be input as a condition in the autoencoder. This can be altered,
# though "has_rec" needs to be included for the experiment to be worthwhile and
# the rest of the script to work. It gives sort of different results depending
# on which information is thrown in, with the best results coming if all data
# is included.

##regress_columns = ["has_rec","rec_x","rec_y"]
regress_columns = columns

assert("has_rec" in regress_columns)
assert(np.all([_ in columns for _ in regress_columns]))

if os.path.isfile(pd_path):
	df = pd.read_pickle(pd_path)
else:
	minw,minh = int(dim[0]/4),int(dim[1]/4)
	maxw,maxh = int(dim[0]/3),int(dim[1]/3)
	data_list = {}
	for i in range(num_random_images):
		imtitle = "im_%.5d" % i
		savepath = os.path.join(image_folder,"%s.png" % imtitle)
		im = Image.new('RGB', dim, (255,255,255))
		draw = ImageDraw.Draw(im)
		has_ell = choice([True,False])
		has_rec = choice([True,False])
		ell_w,ell_h = randint(minw,maxw),randint(minh,maxh)
		ell_x,ell_y = randint(dim[0]-ell_w),randint(dim[1] - ell_h)
		rec_w,rec_h = randint(minw,maxw),randint(minh,maxh)
		rec_x,rec_y = randint(dim[0]-rec_w),randint(dim[1] - rec_h)
		if has_ell:
			draw.ellipse((ell_x,ell_y,ell_x + ell_w,ell_y + ell_h),
				fill=(255,0,0),outline=(0,0,0))
		if has_rec:
			draw.rectangle((rec_x,rec_y,rec_w+rec_x,rec_h+rec_y),
				fill=(0,255,0),outline=(0,0,0))
		im.save(savepath)
		# Scales information to be between 0 and 1. If an ellipses or rectangle
		# is not present, coordinates are there anyway, but the neural network
		# will learn that it's just random numbers, I assume.
		datarow = [	float(rec_x) / dim[0],
			float(rec_y) / dim[1],
			float(rec_w - minw)/(maxw-minw),
			float(rec_h - minh)/(maxh-minh),
			float(has_rec),
			float(ell_x) / dim[0],
			float(ell_y) / dim[1],
			float(ell_w - minw)/(maxw-minw),
			float(ell_h - minh)/(maxh-minh),
			float(has_ell)]
		data_list[savepath] = datarow
	df = pd.DataFrame.from_dict(data_list, columns=columns,orient='index')
	df.to_pickle(pd_path)

# Reads in the data to a numpy array. C_switch is the array that removes the
# green rectangles, setting all of those flags to zero.
X_files = df.index
X = np.zeros((len(X_files),dim[0],dim[1],3))
C = np.zeros((len(X_files),len(regress_columns)))
C_switch = np.zeros((len(X_files),len(regress_columns)))
for i in range(len(X_files)):
	filename = X_files[i]
	im = imageio.imread(filename)
	im = im.astype(float)
	im = im / 255.0
	X[i,:,:,:] = im
	C[i,:] = df.loc[filename,regress_columns]
	C_switch[i,:] = C[i,:]
	C_switch[i,regress_columns.index("has_rec")] = 0

X_train		= X[div:,:,:,:]
X_test		 = X[:div,:,:,:]
C_train		= C[div:,:]
C_test		 = C[:div,:]
C_switch_train = C_switch[div:,:]
C_switch_test  = C_switch[:div,:]


# Construction of the conditional variational autoencoder
# The inclusion of layers like AveragePooling2D/LeakyReLU instead of the usual
# MaxPooling2D/ReLU is not necessary, but I included it because I was using 
# adversarial components in previous iterations, and may reintroduce them in
# the future. They may be replaced if you wish to save on computational power. 

# Inputs for both the image and the labels
input_conf = Input(shape = (len(regress_columns),))
input_img = Input(shape=(dim[0], dim[1], 3))

x = Conv2D(64, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = AveragePooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = AveragePooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Conv2D(32, (4, 4), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)

encoded = AveragePooling2D((2, 2), padding='same')(x)
encoded = Flatten()(encoded)

# I experimented with which part of the encoder to input the labels. It can be
# moved up or down the network. This works reasonably well.
encoded = Concatenate()([encoded,input_conf])
encoded = Dense(256)(encoded)
encoded = LeakyReLU(alpha=0.3)(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(256)(encoded)
encoded = LeakyReLU(alpha=0.3)(encoded)
encoded = BatchNormalization()(encoded)

# The important sampling layers
mu = Dense(n_z, activation='linear')(encoded)
l_sigma = Dense(n_z, activation='linear')(encoded)
samp = Lambda(sample_z,output_shape=(n_z,))([mu,l_sigma])


encoder = Model(inputs = (input_img,input_conf),outputs=(mu,l_sigma,samp))

input_decoder = Input(shape = (samp.shape[1:]))

encoded = Concatenate(axis=1)([input_decoder,input_conf])

encoded = Dense(256)(encoded)
encoded = LeakyReLU(alpha=0.3)(encoded)
encoded = BatchNormalization()(encoded)
encoded = Dense(256)(encoded)
encoded = LeakyReLU(alpha=0.3)(encoded)
encoded = BatchNormalization()(encoded)
encoded = Reshape((1,1,256))(encoded)

x = Conv2D(32, (4, 4), padding='same')(encoded)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = UpSampling2D((4, 4),interpolation='bilinear')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = UpSampling2D((2, 2),interpolation='bilinear')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = UpSampling2D((2, 2),interpolation='bilinear')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = UpSampling2D((2, 2),interpolation='bilinear')(x)
x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)

decoder = Model(inputs = (input_decoder,input_conf),outputs=decoded)
decoder.summary()

# Discriminator model
input_disc = Input(shape = (decoded.shape[1],decoded.shape[2],decoded.shape[3]))
x = Conv2D(64, (3, 3), padding='same')(input_disc)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = AveragePooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = AveragePooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.3)(x)
x = Flatten()(x)
x = Concatenate()([x,input_conf])
x = Dense(256)(x)
x = LeakyReLU(alpha=0.3)(x)
x = BatchNormalization()(x)
x = Dense(256)(x)
x = LeakyReLU(alpha=0.3)(x)
x = BatchNormalization()(x)
x = Dense(256)(x)
x = LeakyReLU(alpha=0.3)(x)
x = BatchNormalization()(x)
disc_output = Dense(1,activation="sigmoid")(x)

# Compile main model
#m = Model(inputs = (input_img,input_conf),outputs = decoded)
#m.compile(loss=vae_loss(l_sigma,mu),optimizer=optimizer)

m = VAE(encoder,decoder)
m.compile(optimizer=optimizer)

if train_discriminator:
	# Compile discriminator
	discriminator = Model(inputs = (input_disc,input_conf),outputs = disc_output)
	discriminator.compile(loss = 'binary_crossentropy',optimizer=optimizer)
	
	# Trains the image generator to fool the discriminator
	discriminator.trainable = False
	combined_model = Model(inputs = (input_img,input_conf),
		outputs=discriminator([decoded,input_conf]))
	combined_model.compile(loss = 'binary_crossentropy',optimizer=optimizer)

start_new_gif = True
j = 0
if check_for_save_and_load_model:
	if os.path.isfile(model_file):
		m = load_model(model_file,
			custom_objects = {'loss': vae_loss(l_sigma,mu)})
	if os.path.isfile(epoch_number_file):
		j = np.load(epoch_number_file)
		start_new_gif = False

while j < max_epochs:
	# Train model
	c_his = m.fit([X_train,C_train],X_train,epochs=1,verbose=verbose,
				batch_size=batch_size)
	if train_discriminator and j % 20 == 0:
	# Train discriminator to tell apart real and fake images
		#true_samples = m.predict([X_train,C_train])
		false_samples = m.predict([X_train,C_switch_train])
		
		discriminator.fit([X_train,C_train],
			np.ones((X_train.shape[0],)),epochs=1,verbose=verbose)
		discriminator.fit([false_samples,C_switch_train],
			np.zeros((X_train.shape[0],)),epochs=1,verbose=verbose)
		combined_model.fit([X_train,C_switch_train],
			np.ones((X_train.shape[0],)),epochs=1,verbose=verbose)
		combined_model.fit([X_train,C_train],
			np.ones((X_train.shape[0],)),epochs=1,verbose=verbose)
	c_loss = np.mean(c_his.history["loss"])
	if verbose: print("%d: C: %.6f" % (j,c_loss))
	if j % 20 == 0 or c_loss <= loss_lim:
		if verbose: print("Saving")
		if check_for_save_and_load_model:
			#m.save(model_file)
			np.save(epoch_number_file,j)
		# Tiles the original, predicted, and rectangle-free images together
		X_test_pred = m.predict([X_test,C_test])
		X_test_pred_switch = m.predict([X_test,C_switch_test])
		for i in range(X_test_pred.shape[0]):
			savefile = os.path.join(image_prediction_folder,"im_%.5d.png" % i)
			im_orig = np.squeeze(X_test[i,:,:,:])
			im = np.squeeze(X_test_pred[i,:,:,:])
			im2 = np.squeeze(X_test_pred_switch[i,:,:,:])
			im = np.concatenate((im_orig,im,im2),axis=1)
			im = (im * 255).astype(np.uint8)
			im = Image.fromarray(im)
			im.save(savefile)
	
		if plot_latent_space:
			_,_,z = encoder.predict([X_test,C_test])
			_,_,z_sw = encoder.predict([X_test,C_switch_test])
			r_col = regress_columns.index("has_rec")
			e_col = regress_columns.index("has_ell")
			has_both = np.squeeze(np.logical_and(C_test[:,r_col] == 1,
												 C_test[:,e_col] == 1))
			has_ell  = np.squeeze(np.logical_and(C_test[:,r_col] == 0,
												 C_test[:,e_col] == 1))
			has_rec  = np.squeeze(np.logical_and(C_test[:,r_col] == 1,
												 C_test[:,e_col] == 0))
			has_none = np.squeeze(np.logical_and(C_test[:,r_col] == 0,
											 C_test[:,e_col] == 0))
			
			plt.scatter(z[has_rec,0],	z[has_rec,1],	label="Rectangle")
			plt.scatter(z[has_ell,0],	z[has_ell,1],	label="Ellipse")
			plt.scatter(z[has_both,0],   z[has_both,1],   label="Both")
			plt.scatter(z_sw[has_both,0],z_sw[has_both,1],
											label="Both (no rect)")
			plt.scatter(z_sw[has_rec,0], z_sw[has_rec,1],
										label="Rectangle (no rect)")
			plt.scatter(z[has_none,0],   z[has_none,1],   label="None")
			plt.legend(loc='upper right')
			plt.title("Epoch %d" % j)
			ax = plt.gca()
			ax.axes.xaxis.set_visible(False)
			ax.axes.yaxis.set_visible(False)

			plt.savefig(latent_space_plot)
			plt.clf()
		for g in gif_list: add_to_gif(g,new_gif=(j==0 and start_new_gif))
	if c_loss <= loss_lim: break
	j += 1
