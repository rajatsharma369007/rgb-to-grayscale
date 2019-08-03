import tensorflow as tf
import numpy as np
import cv2

num_images = 3670

dataset = []

for i in range(1, num_images+1):
    img = cv2.imread("data/color_images/color_" +str(i) +".jpg" )
    dataset.append(np.array(img))

dataset_source = np.asarray(dataset)
print(dataset_source.shape)

dataset_tar = []

for i in range(1, num_images+1):
    img = cv2.imread("data/gray_images/gray_" +str(i) +".jpg", 0)    
    dataset_tar.append(np.array(img))

dataset_target = np.asarray(dataset_tar)
print(dataset_target.shape)

dataset_target = dataset_target[:, :, :, np.newaxis]

def autoencoder(inputs): # Undercomplete Autoencoder
    
    # Encoder
    
    net = tf.layers.conv2d(inputs, 128, 2, activation = tf.nn.relu)
    print(net.shape)
    net = tf.layers.max_pooling2d(net, 2, 2, padding = 'same')
    print(net.shape)

    # Decoder
    
    net = tf.image.resize_nearest_neighbor(net, tf.constant([129, 129]))
    net = tf.layers.conv2d(net, 1, 2, activation = None, name = 'outputOfAuto')

    print(net.shape)
    
    return net

ae_inputs = tf.placeholder(tf.float32, (None, 128, 128, 3), name = 'inputToAuto')
ae_target = tf.placeholder(tf.float32, (None, 128, 128, 1))

ae_outputs = autoencoder(ae_inputs)
lr = 0.001

loss = tf.reduce_mean(tf.square(ae_outputs - ae_target))
train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
# Intialize the network 
init = tf.global_variables_initializer()

batch_size = 32
epoch_num = 50

saving_path = 'model/ColorToGray.ckpt'

saver_ = tf.train.Saver(max_to_keep = 3)

batch_img = dataset_source[0:batch_size]
batch_out = dataset_target[0:batch_size]

num_batches = num_images//batch_size

sess = tf.Session()
sess.run(init)

for ep in range(epoch_num):
    batch_size = 0
    for batch_n in range(num_batches): # batches loop

        _, c = sess.run([train_op, loss], feed_dict = {ae_inputs: batch_img, ae_target: batch_out})
        print("Epoch: {} - cost = {:.5f}" .format((ep+1), c))
            
        batch_img = dataset_source[batch_size: batch_size+32]
        batch_out = dataset_target[batch_size: batch_size+32]
            
        batch_size += 32
    
    saver_.save(sess, saving_path, global_step = ep)
recon_img = sess.run([ae_outputs], feed_dict = {ae_inputs: batch_img})

sess.close()

saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver.restore(sess, 'model/ColorToGray.ckpt-49')

import glob as gl 

filenames = gl.glob('data/input_images/*.jpg')

test_data = []
for file in filenames[0:100]:
    test_data.append(np.array(cv2.imread(file)))

test_dataset = np.asarray(test_data)
print(test_dataset.shape)

# Running the test data on the autoencoder
batch_imgs = test_dataset
gray_imgs = sess.run(ae_outputs, feed_dict = {ae_inputs: batch_imgs})

print(gray_imgs.shape)

for i in range(gray_imgs.shape[0]):
    cv2.imwrite('data/output_images/' +str(i) +'.jpg', gray_imgs[i])






