# giefbot.py - Rosie Kerwin, Joshua Rappaport, Manickam Manickam
# 11.28.2017

import tensorflow as tf
import gym
import numpy as np
import random as rand
import operator
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
    rand.seed()
    env = gym.make('Asteroids-v0')
    observation = env.reset()
    #observation = downsample(observation)
    #reward = 0
    action = 0
    env.render()
    prev_obs = []
    curr_obs = []
    D = []
    sess, output_net, x = initialize()
    for i in range(4):
        observation, reward, done, info = env.step(action)  # pass in 0 for action
        observation = convert_to_small_and_grayscale(observation)
        prev_obs = curr_obs
        curr_obs = obsUpdate(curr_obs,observation)
        e = [reward, action, prev_obs, curr_obs]
        D.append(e)
        action = 0
        
    for i in range(1000):
        if (len(D) > 256):
            D.pop()
        action = magic(curr_obs, sess, output_net, x) #change this to just take in curr_obs, sess, and False
        env.render()
        observation, reward, done, info = env.step(action) # take a random action
        observation = convert_to_small_and_grayscale(observation)
        e = [reward, action, prev_obs, curr_obs]
        D.append(e)
        prev_obs = curr_obs
        curr_obs = obsUpdate(curr_obs,observation)
        update_q_function(D, sess)
            
def update_q_function(D, sess):
    gamma = 0.99
    T = 4
    indexes = []
    for _ in range(T):
        indexes.append(rand.randrange(len(D)))
    
    for e in indexes:
        rt = e[0]
        at = e[1]
        curr_obs = e[2]
        future_obs = e[3]
        nQ = sess.run(x, feed_dict={x: future_obs})
        m = [0]*6
        m[at] = 1
        sess.run(trainer, feed_dict={mask: m, reward: rt, currentQ: curr_obs, nextQ: future_obs})

def obsUpdate(curr_obs,new_obs):
    if len(curr_obs) > 3:
        curr_obs.pop()
    curr_obs.insert(0,new_obs)
    return curr_obs

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(rshape):
  return tf.Variable(tf.constant(0.1, shape=rshape))

def conv2d(x, W, st):
  return tf.nn.conv2d(x, W, strides=[1, st, st, 1], padding='SAME')

def initialize():
    sess = tf.Session()
    x = tf.placeholder(tf.float32, shape = [4,84,84])
    #x = tf.placeholder(tf.float32, shape = [84,84,4])

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
    conv_out = tf.reshape(h_conv3, [-1, 11*11*64]) # this is the line that was giving us issues.

    w_hidden = weight_variable([7744,512])
    b_hidden = bias_variable([512])
    hidden_net = tf.matmul(conv_out,w_hidden)+b_hidden
    hidden_out = tf.nn.relu(hidden_net)

    w_output = weight_variable([512,6])
    b_output = bias_variable([6])
    output_net = tf.matmul(hidden_out,w_output)+b_output
    
    currentQ = tf.Variable(tf.constant(0.1, shape=[6]))
    reward = tf.Variable(tf.constant(0.1, shape=[6]))
    nextQ = tf.Variable(tf.constant(0.1, shape=[6]))
    mask = tf.Variable(tf.constant(0.1, shape=[6]))
    cost = (((mask*reward)+(mask*nextQ))-mask*currentQ)
    print(type(cost))
    #print(tf.trainable_variables())
    trainer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    sess.run(tf.global_variables_initializer())
    return sess, output_net, x

def magic(curr_obs, sess, output_net, x):
    print(curr_obs)
    var = sess.run(output_net, feed_dict={x: curr_obs})
    prediction_index, predicted_value = max(enumerate(var), key=operator.itemgetter(1))
    print(prediction_index)
    return prediction_index



if (__name__ == "__main__"):
        main()
