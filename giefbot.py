# giefbot.py - Rosie Kerwin, Joshua Rappaport, Manickam Manickam
# 11.28.2017

import tensorflow as tf
import gym
import numpy as np
import random as rand
import operator
from copy import deepcopy
#import cv2

num_actions = 4

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
    #print(len(ret[1]))
    
    
    #print(ret.shape)
    return ret

def main():
    rand.seed()
    #env = gym.make('Asteroids-v0')
    env = gym.make('Breakout-v0')
    num_actions = env.action_space.n
    print(num_actions)
    observation = env.reset()
    #observation = downsample(observation)
    #reward = 0
    action = 0
    env.render()
    prev_obs = []
    curr_obs = []
    D = []
    step = 0
    rate = 1
    sess, output_net, x, cost, trainer, mask, reward, nextQ = initialize()
    #load(sess)
    startPrinting = False
    for i in range(5):
        observation, rw, done, info = env.step(action)  # pass in 0 for action
        observation = convert_to_small_and_grayscale(observation)
        prev_obs = deepcopy(curr_obs)
        curr_obs = obsUpdate(curr_obs,observation)
        #e = [rw, action, deepcopy(prev_obs), deepcopy(curr_obs)]
        #D.append(e)
        action = 0
        #print(i)
    print("Entering mini-loop")
    for _ in range(10):
        step +=1
        #print(step)
        if done:
            observation = env.reset()
        if (len(D) > 256):
            D.pop()
        if step % 1000 == 0:
            rate = rate / 2
        #if step % 1000 == 0:
            #save(sess)
        action = magic(curr_obs, sess, output_net, x,step,rate, False) #change this to just take in curr_obs, sess, and False
        #action = env.action_space.sample()
        env.render()
        observation, rw, done, info = env.step(action) # take a random action
        #print(action, rw, step)
        observation = convert_to_small_and_grayscale(observation)
        e = [rw, action, deepcopy(prev_obs), deepcopy(curr_obs)]
        D.append(e)
        prev_obs = deepcopy(curr_obs)
        curr_obs = obsUpdate(curr_obs,observation)
    print("Entering full loop")
    while True:
        step +=1
        
        #print(step)
        if done:
            observation = env.reset()
        if (len(D) > 500):
            D.pop()
        if step % 100 == 0:
            print(step,"steps have passed")
            save(sess)
        if step % 100 == 0:
            rate = rate / 2
            startPrinting = True
            #print(step,"steps have passed")
        #if step % 1000 == 0:
            #save(sess)
        action = magic(curr_obs, sess, output_net, x,step,rate, startPrinting) #change this to just take in curr_obs, sess, and False
        #action = env.action_space.sample()
        env.render()
        observation, rw, done, info = env.step(action) # take a random action
        #print(action, rw, step)
        observation = convert_to_small_and_grayscale(observation)
        e = [rw, action, deepcopy(prev_obs), deepcopy(curr_obs)]
        D.append(e)
        prev_obs = deepcopy(curr_obs)
        curr_obs = obsUpdate(curr_obs,observation)
        update_q_function(D, sess, output_net, x, cost, trainer, mask, reward, nextQ)

def save(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/aster.ckpt")
    print("Model saved in file: %s" % save_path)

def load(sess):
    saver = tf.train.Saver()
    saver.restore(sess, "./model/aster.ckpt")
    print("Model restored.")

def update_q_function(D, sess, output_net, x, cost, trainer, mask, reward, nextQ):
    gamma = 0.99
    T = 4
    indexes = []
    for _ in range(T):
        indexes.append(D[rand.randrange(len(D))])
    
    for e in indexes:
        rt = [e[0]]
        at = e[1]
        curr_obs = e[2]
        future_obs = e[3]
        nQ = sess.run(output_net, feed_dict={x: future_obs})
        nQ = [np.max(nQ)]
        m = [0]*num_actions
        m[at] = 1
        #print(m, rt, curr_obs, future_obs, "****")
        #_, c, p = sess.run([trainer, cost, output_net], feed_dict={mask: m, reward: rt, x: curr_obs, nextQ: nQ})
        _ = sess.run([trainer], feed_dict={mask: m, reward: rt, x: curr_obs, nextQ: nQ})
        #print(nQ, p)
        #print(c)

def obsUpdate(curr_obs,new_obs):
    #print("STARTING OBSUPDATE")
    if (len(curr_obs) == 0):
        #print("MAKING NEW CURR_OBS OBJECT")
        curr_obs = []
        for x in range(84):
            curr_obs.append([])
            for y in range(84):
                curr_obs[x].append([])
        for i in range(len(curr_obs)):
            for j in range(len(curr_obs[i])):
                item = []
                item.append(deepcopy(new_obs[i][j]))
                #print(item)
                curr_obs[i][j] = item
        #print(len(curr_obs[1][2]), len(curr_obs[1]))
        return curr_obs
    #print(len(curr_obs[2][12]), curr_obs[2][12])
    if len(curr_obs[0][0]) > 3:
        #print("deleting 1 screen")
        for i in range(len(curr_obs)):
            for j in range(len(curr_obs[i])):
                curr_obs[i][j].pop()
    
    #print(len(curr_obs[2][12]), curr_obs[2][12], new_obs[2][12])
    #print(type(new_obs[2][12]), type(curr_obs[2][12][0]))
    #print(new_obs.shape)
    for i in range(len(curr_obs)):
        for j in range(len(curr_obs[i])):
            item = new_obs[i][j]
            #print(item)
            
            curr_obs[i][j].insert(0, item)
    #print(len(curr_obs[2][12]), curr_obs[2][12])
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
    #x = tf.placeholder(tf.float32, shape = [4,84,84])
    x = tf.placeholder(tf.float32, shape = [84,84,4], name="insert")

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

    w_output = weight_variable([512,num_actions])
    b_output = bias_variable([num_actions])
    output_net = tf.matmul(hidden_out,w_output)+b_output
    
    #currentQ = tf.Variable(tf.constant(0.1, shape=[6]))
    reward = tf.placeholder(tf.float32, shape = [1])
    nextQ = tf.placeholder(tf.float32, shape = [1])
    mask = tf.placeholder(tf.float32, shape = [num_actions])
    cost = (((mask*reward)+(mask*nextQ))-mask*output_net)**2
    #print(type(cost))
    #print(tf.trainable_variables())
    trainer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    sess.run(tf.global_variables_initializer())
    return sess, output_net, x, cost, trainer, mask, reward, nextQ

def magic(curr_obs, sess, output_net, x,step,rate, startPrinting):
    #print(curr_obs)
    var = sess.run(output_net, feed_dict={x: curr_obs})
    #prediction_index, predicted_value = max(enumerate(var), key=operator.itemgetter(1))
    prediction_index = np.argmax(var)
    #print(var, prediction_index)
    
    
    #uncomment below to return the randomly predicted actions
    r = rand.random()
    if r < rate:
        vals = list(range(num_actions))
        vals.remove(prediction_index)
        prediction_index = vals[rand.randint(0, len(vals) - 1)]
        if startPrinting:
            print("Random action chosen: ", prediction_index)
            print(var)
    else:
        if startPrinting:
            print("Networked action chosen: ", prediction_index)
            print(var)
        
    #print(prediction_index)
    return prediction_index



if (__name__ == "__main__"):
        main()
