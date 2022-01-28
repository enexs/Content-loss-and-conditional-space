# This code block is tensorflow implementation of Pix2Pix GAN model
# and equivalent to original PyTorch implementation presented at
# Image-to-Image Translation with Conditional Adversarial Networks
# Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
# CVPR, 2017.
# https://github.com/phillipi/pix2pix
# pix2pix gan for satellite to map image-to-image translation borrowed from
# https://machinelearningmastery.com
#
#
# Different than the original implementation here we used two sequential generators
# trained with two discriminators


from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
# from tensorflow.keras.models import Input
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model
import scipy.io as sio


# define the discriminator model
def define_discriminator(image_shape, name):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out, name=name)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g


# define the standalone generator model
def define_generator(image_shape=(256, 256, 3), name='name'):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image, name=name)
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model_1, g_model_2, d_model_1, d_model_2, image_shape):
    # make weights in the discriminator not trainable
    # d_model.trainable = False eski versiyon
    # make weights in the discriminator not trainable
    for layer in d_model_1.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    for layer in d_model_2.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator_1 input
    gen_1_out = g_model_1(in_src)
    # connect the gen_1_out image to the disc_1 input
    dis_out_1 = d_model_1([in_src, gen_1_out])
    # connect the gen_1_out image to the generator_2 input
    gen_2_out = g_model_2(gen_1_out)

    # connect the source input and generator output to the discriminator input
    dis_out_2 = d_model_2([in_src, gen_2_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out_1, dis_out_2, gen_1_out, gen_2_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    plot_model(model, to_file='gan_model.png', show_shapes=True)
    model.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'mae', 'mae'], optimizer=opt,
                  loss_weights=[1, 1, 100, 100])
    return model


# load and prepare training images
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model_1, g_model_2, samples, patch_shape):
    # generate fake instance
    X_fakeB_1 = g_model_1.predict(samples)
    X_fakeB_2 = g_model_2.predict(X_fakeB_1)
    # create 'fake' class labels (0)
    y = zeros((len(X_fakeB_1), patch_shape, patch_shape, 1))
    return X_fakeB_1, X_fakeB_2, y


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model_1, g_model_2, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB_1, X_fakeB_2, _ = generate_fake_samples(g_model_1, g_model_2, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB_1 = (X_fakeB_1 + 1) / 2.0
    X_fakeB_2 = (X_fakeB_2 + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image 1
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB_1[i])
    # plot generated target image 1
    # plot generated target image 1
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB_2[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(4, n_samples, 1 + n_samples * 3 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_1_%06d.h5' % (step + 1)
    filename3 = 'model_2_%06d.h5' % (step + 1)
    g_model_1.save(filename2)
    g_model_2.save(filename3)
    print('>Saved: %s and %s and %s' % (filename1, filename2, filename3))


# train pix2pix models
def train(d_model_1, d_model_2, g_model_1, g_model_2, gan_model, dataset, n_epochs=100, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model_1.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    L1lossList_1 = []
    L1lossList_2 = []
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB_1, X_fakeB_2, y_fake = generate_fake_samples(g_model_1, g_model_2, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1_1 = d_model_1.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss1_2 = d_model_1.train_on_batch([X_realA, X_fakeB_1], y_fake)

        # update discriminator for real samples
        d_loss2_1 = d_model_2.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2_2 = d_model_2.train_on_batch([X_realA, X_fakeB_2], y_fake)

        # update the generator
        g_loss, discrLoss1, discrLoss2, L1loss_1, L1loss_2 = gan_model.train_on_batch(X_realA, [y_real, y_real, X_realB,
                                                                                                X_realB])
        # summarize performance
        L1lossList_1.append(L1loss_1)
        L1lossList_2.append(L1loss_2)
        print(
            '>%d, d1_1[%.3f] d1_2[%.3f] d2_1[%.3f] d2_2[%.3f] g_loss[%.3f] discrLoss1[%.3f] discrLoss2[%.3f] '
            'L1loss_1[%.3f] L1loss_2[%.3f]' %
            (i + 1, d_loss1_1, d_loss1_2, d_loss2_1, d_loss2_2, g_loss, discrLoss1, discrLoss2, L1loss_1, L1loss_2))
        # summarize model performance
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model_1, g_model_2, dataset)
    # print(L1lossList)
    file_name_for_1 = "L1_loss_file_2gen_2disc/L1_loss_2generator1.mat"
    file_name_for_2 = "L1_loss_file_2gen_2disc/L1_loss_2generator2.mat"
    sio.savemat(file_name_for_1, {'vect': L1lossList_1})
    sio.savemat(file_name_for_2, {'vect': L1lossList_2})


# load image data
dataset = load_real_samples('maps_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]

# define the models
d_model_1 = define_discriminator(image_shape, name='disc1')
d_model_1.summary()
plot_model(d_model_1, to_file='d_model_1.png', show_shapes=True)

# define the models
d_model_2 = define_discriminator(image_shape, name='disc2')
d_model_2.summary()
plot_model(d_model_2, to_file='d_model_2.png', show_shapes=True)

# generator_1
g_model_1 = define_generator(image_shape, name='gen1')
g_model_1.summary()
plot_model(g_model_1, to_file='g_model_1.png', show_shapes=True)

# generator_2
g_model_2 = define_generator(image_shape, name='gen2')
g_model_2.summary()
plot_model(g_model_2, to_file='g_model_2.png', show_shapes=True)

# define the composite model
gan_model = define_gan(g_model_1, g_model_2, d_model_1, d_model_2, image_shape)
gan_model.summary()
# plot_model(gan_model, to_file='gan_model.png', show_shapes=True)

# train model
train(d_model_1, d_model_2, g_model_1, g_model_2, gan_model, dataset)
