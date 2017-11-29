# giefbot.py - Rosie Kerwin, Joshua Rappaport, Manickam Manickam
# 11.28.2017

def network(train, test, numAttr, numLabels, NUM_NEURONS, LEARNING_RATE, iterations):
    #build attr tensor
    x = tf.placeholder(tf.float32, shape = [None, numAttr])
    
    #create hidden layer
    W_hidden = tf.Variable(tf.truncated_normal([numAttr, NUM_NEURONS], stddev = 0.1))
    b_hidden = tf.Variable(tf.constant(0.1, shape=[NUM_NEURONS]))
    
    net_hidden = tf.matmul(x, W_hidden) + b_hidden
    out_hidden = tf.sigmoid(net_hidden)

    #create output layer
    W_output = tf.Variable(tf.truncated_normal([NUM_NEURONS, numLabels], stddev = 0.1))
    b_output = tf.Variable(tf.constant(0.1, shape=[numLabels]))
    
    net_output = tf.matmul(out_hidden, W_output) + b_output
    
    out_hidden = tf.sigmoid(out_hidden)

    #create true labels
    y = tf.placeholder(tf.float32, shape=[None, numLabels])
   
    #create training
    if numLabels == 1:
        predict = tf.sigmoid(net_output)
    else: 
        predict = tf.nn.softmax(net_output)
    
    #create training
    if numLabels == 1:
        cost = tf.reduce_sum(0.5 * (y-predict) * (y-predict))
    else:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=net_output))
    
    trainer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost) #backprop

    #start tf session
    sess = tf.Session()
    init = tf.initialize_all_variables().run(session=sess)
    
    #train
    step = 0
    maxSteps = iterations
    
    while (step < maxSteps):
        step += 1
        _, p = sess.run([trainer, predict], feed_dict={x: train[0], y: train[1]})
        t = sess.run(predict, feed_dict={x: test[0]})
        if step % 50 == 0:
            print "training"
            getAccs(p, train)
            print "test"
            getAccs(p, test)
    p = sess.run(predict, feed_dict={x: test[0]})
    return p, test