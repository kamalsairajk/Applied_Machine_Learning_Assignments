# -*- coding: utf-8 -*-
""" CIS4930/6930 Applied ML --- nets.py
"""

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
    
from tensorflow.keras.layers import Input, Flatten, Reshape, Dense, Dropout, LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization


# Custom layer for VAE 
class VAESampleLayer(tf.keras.layers.Layer):
    
    # implement call() which will sample gaussians with params mu and sigma^2
    def call(self, in_mu_log_sigma2):
        mu, log_sigma2 = in_mu_log_sigma2 # unpack the input
        stddev = tf.exp(log_sigma2/2.0)
       
        # produce standard normal samples and then add mu and scale by stddev
        return mu + tf.random.normal(tf.shape(mu)) * stddev
    
## VAE
def create_simple_vae(input_shape=(28, 28), latent_width=32, hidden_widths=[200, 100], kldiv_weight=0.1, verbose=False):
    # encoder
    enc_input = Input(shape=input_shape, name='Encoder-input')
    enc_flatten = Flatten(name='Flatten')(enc_input)
    enc_fc1 = Dense(hidden_widths[0], activation='relu', name='Encoder-FC1')(enc_flatten)
    enc_fc2 = Dense(hidden_widths[1], activation='relu', name='Encoder-FC2')(enc_fc1)
    
    # create the latent space repr
    enc_latent_repr_mu = Dense(latent_width)(enc_fc2) # the mean mu
    enc_latent_repr_log_sigma2 = Dense(latent_width)(enc_fc2) # this will be the log of the variance (sigma^2)
    
    # VAE sampling
    vae_sample = VAESampleLayer()
    
    # connect it by passing it the inputs
    enc_latent_repr = vae_sample((enc_latent_repr_mu, enc_latent_repr_log_sigma2))
    
    # define the model with three outputs, the latent repr, mu, and log of sigma^2
    enc_outputs = [enc_latent_repr, enc_latent_repr_mu, enc_latent_repr_log_sigma2]
    enc_model = keras.Model(inputs=[enc_input], outputs=enc_outputs, name='Encoder')
    
    # decoder
    dec_input = Input(shape=[latent_width], name='Decoder-input')
    dec_fc1 = Dense(hidden_widths[1], activation='relu', name='Decoder-FC1')(dec_input)
    dec_fc2 = Dense(hidden_widths[0], activation='relu', name='Decoder-FC2')(dec_fc1)
    dec_output = Dense(input_shape[0]*input_shape[1], activation='sigmoid', name='Decoder-output')(dec_fc2)
    dec_reshape = Reshape(input_shape)(dec_output)
    dec_model = keras.Model(inputs=[dec_input], outputs=[dec_reshape], name='Decoder')
    
    # connect encoder and decoder
    latent_space_output = enc_model(enc_input)
    latent_space_repr = latent_space_output[0] # we feed only 'latent_space_repr' to the decoder
    reconstructed_output = dec_model(latent_space_repr)
    
    vae_model = keras.Model(inputs=[enc_input], outputs=[reconstructed_output], name='VAE-simple')
    

    # we need to add the latent-space KL-div loss to the reconstruction loss 
    # we could make a custom loss, but we can just use keras' add_loss()
    
    # see: https://arxiv.org/pdf/1606.05908.pdf for details
    kldiv_loss = -1.0/2.0 * tf.reduce_mean(1.0 - tf.square(enc_latent_repr_mu) + enc_latent_repr_log_sigma2 - 
                                          tf.math.exp(enc_latent_repr_log_sigma2))
    vae_model.add_loss(kldiv_weight * kldiv_loss)
    
    if verbose:
        vae_model.summary()
    
    opt = keras.optimizers.Adam(lr=0.002)
    vae_model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=opt, metrics=['binary_crossentropy', 'mse']) 

    return vae_model, enc_model, dec_model
    


def create_gan(output_dim=28*28, latent_width=64, verbose=False):

    # generator
    gen = keras.models.Sequential(name='Generator')
    
    in_layer = Input(shape=(latent_width,), name='gen-input')
    gen.add(in_layer)
    
    num_units = 7 * 7 * 128
    gen.add(Dense(num_units, name='gen-fc1'))
    gen.add(LeakyReLU(0.2))
    
    gen.add(Reshape((7, 7, 128)))

    # go from 7x7 to 14x14
    gen.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', name='gen-deconv1'))
    gen.add(LeakyReLU(0.2))
    gen.add(BatchNormalization(name='gen-bn1'))
    
    # go from 14x14 to 28x28
    gen.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', name='gen-deconv2'))
    gen.add(LeakyReLU(0.2))
    gen.add(BatchNormalization(name='gen-bn2'))
    
    gen.add(Conv2D(1, (7,7), padding='same', activation='sigmoid', name='gen-output-conv'))


    # discriminator
    discr = keras.models.Sequential(name='Discriminator')
    
    discr.add(Input(shape=[28, 28, 1], name='discr-input'))
    
    discr.add(Conv2D(64, (3,3), strides=(2,2), padding='same', activation=LeakyReLU(0.2), name='discr-conv1'))
    discr.add(Dropout(0.3))
    
    discr.add(Conv2D(64, (3,3), strides=(2,2), padding='same', activation=LeakyReLU(0.2), name='discr-conv2'))
    discr.add(Dropout(0.3))

    discr.add(Flatten())
    discr.add(Dense(1, activation='sigmoid', name='discr-output'))
  

    # connect the two
    discr_in = gen(in_layer)
    output = discr(discr_in)

    gan = keras.Model(inputs=[in_layer], outputs=[output], name='GAN')
    
    discr.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0004, beta_1=0.5))
    discr.trainable = False
    
    # compile
    gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0004, beta_1=0.5))
    
    if verbose:
        gan.summary()
    
    return gan, gen, discr
