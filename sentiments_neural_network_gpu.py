import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import tensorflow as tf

from create_sentiment_featuresets import create_feature_sets_and_labels

#from tensorflow.examples.tutorials.mnist import input_data
import pickle
import numpy as np

train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

n_classes = 2
batch_size = 50
hm_epochs = 10

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

W1 = tf.Variable(tf.random.normal([len(train_x[0]), n_nodes_hl1]))
B1 = tf.Variable(tf.random.normal([n_nodes_hl1]))

W2 = tf.Variable(tf.random.normal([n_nodes_hl1, n_nodes_hl2]))
B2 = tf.Variable(tf.random.normal([n_nodes_hl2]))

W3 = tf.Variable(tf.random.normal([n_nodes_hl2, n_nodes_hl3]))
B3 = tf.Variable(tf.random.normal([n_nodes_hl3]))

W4 = tf.Variable(tf.random.normal([n_nodes_hl3, n_classes]))
B4 = tf.Variable(tf.random.normal([n_classes]))



tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder('float')
y = tf.compat.v1.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight': W1,
                  'bias':B1}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':W2,
                  'bias':B2}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':W3,
                  'bias':B3}

output_layer = {'f_fum':None,
                'weight': W4,
                'bias': B4,}


# Nothing changes
def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x):
	prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	# NEW:
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.optimizers.Adam(learning_rate=0.001).minimize(cost,var_list=[W1,B1,W2,B2,W3,B3,W4,B4])

	with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
		sess.run(tf.global_variables_initializer())
	    
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})
				epoch_loss += c
				i+=batch_size
				
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

	    
train_neural_network(x)