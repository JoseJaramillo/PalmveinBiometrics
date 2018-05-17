'''
Convolutional NN for plamvein classification
accuracy of 50.5% obtained training with 8 images of each person (50)
classifying 4 img of each. (101/200) correctly classified.
'''

import Database
import convolutional
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta
# Convolutional Layer 1.
filter_size1=5                  # Convolution filters are 5x5
num_filters1=36                 # We have 16 of these filters

# Convolutional Layer 2.
filter_size2=5                  # Convolution filters are 5x5
num_filters2=36                 # We have 36 of these filters

# Multilayer perceptron
fc_size= 128                    #number of imput neurons in ml perceptron
def getnewbatch(iteraction):
    global batch_iterations
    miniX=trainX[(25*batch_iterations):(25+(25*batch_iterations))]
    miniY=trainY[(25*batch_iterations):(25+(25*batch_iterations))]
    batch_iterations+=1
    if batch_iterations == 16:
        batch_iterations=0
#    print('X:')
#    print(len(miniX))
#    print('Y:')
#    print(len(miniY))
    return miniX,miniY
def  plot_images (images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations


    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch,y_true_batch= getnewbatch(total_iterations)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)


        # Print status every 100 iterations.
        if i % 50 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))
            print_test_accuracy()
    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
# Split the test-set into smaller batches of this size.
test_batch_size = 25
def print_test_accuracy():

    # Number of images in the test-set.
    num_test = len(testX)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = testX[i:j, :]

        # Get the associated labels.
        labels = testY[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true =np.zeros(200,np.int8)
    for i in range(0,len(testY)):
        cls_true[i] = np.argmax(testY[i])

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
#    print(correct)
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
#    if show_example_errors:
#        print("Example errors:")
#        plot_example_errors(cls_pred=cls_pred, correct=correct)
#
#    # Plot the confusion matrix, if desired.
#    if show_confusion_matrix:
#        print("Confusion Matrix:")
#        plot_confusion_matrix(cls_pred=cls_pred)
#from tensorflow.examples.tutorials.mnist import input_data
#data = input_data.read_data_sets('data/MNIST/', one_hot=True)

#print("Size of:")
#print("- Training-set:\t\t{}".format(len(data.train.labels)))
#print("- Test-set:\t\t{}".format(len(data.test.labels)))
#print("- Validation-set:\t{}".format(len(data.validation.labels)))
#
#xx= data.train.labels
############################IMPORT DATA##################################
if 'trainX' in globals():
    print("Database already extracted")
else:
    
    # Extract Palm images from the Left and use 2/3 sets for training
    # 1/3 remaining will be used as test
    
    print("Extracting Database...")
    trainX,trainY,testX,testY=Database.extractdatabase('Palm','Left',2)
    print("Database extracted!")

###########################!IMPORT DATA!#################################
#data.test.cls = np.argmax(data.test.labels, axis=1)
#yy=data.test.cls
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = len(testX[0])

# Tuple with height and width of images used to reshape arrays.
img_shape = (48,64)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 50


# Get the first images from the test-set.
images = testX[0:9]

# Get the true classes for those images.
cls_true = np.argwhere(testY[0:9])[:,1]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, 48, 64, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = \
    convolutional.new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = \
    convolutional.new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
layer_flat, num_features = convolutional.flatten_layer(layer_conv2)
layer_fc1 = convolutional.new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc2 = convolutional.new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

# It computes the softmax internally that is why we use layer_fc2 instead of y_pred
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

#cross_entropy = K.switch(tf.size(y_true) > 0,
#                    tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1),
#                    tf.constant(0.0))
cost = tf.reduce_mean(cross_entropy)

optimizer  = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

#   Batch size will be constant to 25

# Counter for total number of iterations performed so far.
total_iterations = 0
batch_iterations = 0

##############################################
optimize(num_iterations=5000)
#convolutional.print_test_accuracy(show_example_errors=True)
##############################################


# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
# session.close()
