# giefbot.py - Rosie Kerwin, Joshua Rappaport, Manickam Manickam
# 11.28.2017

import tensorflow as tf
import gym
import numpy as np
#import cv2

# NOTE: RETURNS A SHAPE OF 84x84.
def convert_to_small_and_grayscale(rgb):
    ret = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    shape = ret.shape
    horiz = range(8, shape[0], 2)
    vert = range(8, shape[1], 2)
    top = range(0, 25)
    ret = np.delete(ret, vert, 1)
    ret = np.delete(ret, horiz, 0)
    ret = np.delete(ret, top, 0)
    print(ret.shape)
    return ret

def main():
    env = gym.make('Asteroids-v0')
    observation = env.reset()
    #observation = downsample(observation)
    reward = 0
    action = 0
    prev_obs = []
    curr_obs = [observation]
    observation, reward, done, info = env.step(action)  # pass in 0 for action

    env.render()
    last4obs = []
    numSavedInstances = 0
    
    for i in range(1000):
        prev_obs = curr_obs
        curr_obs = obsUpdate(curr_obs,observation)
        # action = magic(reward,action,prev_obs,curr_obs)
        env.render()
        observation, reward, done, info = env.step(action) # take a random action
        observation = convert_to_small_and_grayscale(observation)
        #print(len(observation))
        #observation = downsample(observation)
            
     
def obsUpdate(curr_obs,new_obs):
    if len(curr_obs) > 3:
        curr_obs.pop()
    curr_obs.insert(0,new_obs)
    return curr_obs

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, st):
  return tf.nn.conv2d(x, W, strides=[1, st, st, 1], padding='SAME')


def magic(reward,action,prev_obs,curr_obs):
    #l1Neurons = 33600
    #build attr tensor
    x = tf.placeholder(tf.float32, shape = [84, 84, 4])

    actions = tf.Variable(tf.truncated_normal([l1Neurons, 5], stddev = 0.1))

    W_conv1 = weight_variable([8,8,4,32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x,[-1,84,84,4])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1, 4)+b_conv1)

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    print(h_conv3.get_shape)
    conv_out = tf.reshape(h_conv3, [-1, 84, 84, 64])
    
    
    
    # w_connected = weight_variable()
    # b_connected = bias_variable([512])
    # h_connected = tf.nn.relu()


    # #create hidden layer
    # W_hidden = tf.Variable(tf.truncated_normal([numAttr, NUM_NEURONS], stddev = 0.1))
    # b_hidden = tf.Variable(tf.constant(0.1, shape=[NUM_NEURONS]))
    #
    # net_hidden = tf.matmul(x, W_hidden) + b_hidden
    # out_hidden = tf.sigmoid(net_hidden)
    #
    # #create output layer
    # W_output = tf.Variable(tf.truncated_normal([NUM_NEURONS, numLabels], stddev = 0.1))
    # b_output = tf.Variable(tf.constant(0.1, shape=[numLabels]))
    #
    # net_output = tf.matmul(out_hidden, W_output) + b_output
    #
    # out_hidden = tf.sigmoid(out_hidden)
    #
    # #create true labels
    # y = tf.placeholder(tf.float32, shape=[None, numLabels])
    #
    # #create training
    # if numLabels == 1:
    #     predict = tf.sigmoid(net_output)
    # else:
    #     predict = tf.nn.softmax(net_output)
    #
    # #create training
    # if numLabels == 1:
    #     cost = tf.reduce_sum(0.5 * (y-predict) * (y-predict))
    # else:
    #     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net_output))
    #
    # trainer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost) #backprop
    #
    # #start tf session
    # sess = tf.Session()
    # init = tf.initialize_all_variables().run(session=sess)
    #
    # #train
    # step = 0
    # maxSteps = iterations
    #
    # while (step < maxSteps):
    #     step += 1
    #     _, p = sess.run([trainer, predict], feed_dict={x: train[0], y: train[1]})
    #     t = sess.run(predict, feed_dict={x: test[0]})
    #     if step % 50 == 0:
    #         print "training"
    #         getAccs(p, train)
    #         print "test"
    #         getAccs(p, test)
    # p = sess.run(predict, feed_dict={x: test[0]})
    # return p, test



if (__name__ == "__main__"):
        main()