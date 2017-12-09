"""
Artificial Intelligence & Robotics
Biometric systems
Palmvein recognition / Deep Neural Networks
Jose Vicente Jaramillo

*** Objetive ***

Having 12 palmvein images for each person of the same hand, we train the 
network with 8 probes (taken in two sessions one week appart) and try to
get the subject corresponding to the 4 resting probes (taken one week
appart).

*** Results ***

After 2000 epochs we have 8% accuracy after 10000 epochs no improvement 
was noticed. It can be improved by a more specific feature extraction, 
by adding more hidden layers, and by adding a method to avoud overfiting 
like dropout.

"""
#Library implemented to extract data from the Dataset
import Database
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if 'trainx' in locals():
    print('Database is already extracted')
else:
    print('Extracting database...')
    trainx,trainy,testx,testy=Database.extractdatabase('Palm','Left',2)
    print('Database extracted!')
    
#NeuralNetwork parameters
inputN=len(trainx[0])
hiddenN=[1000,200]
outputN=50
learningRate=0.001
trainingEpochs=10000
batchSize=200
t=0
ttt=np.zeros(1)
nnnnnn=np.zeros(1)
print("HiddenLayer1:",hiddenN[0],"HiddenLayer2:",hiddenN[1])

#Tensors reserved for the dataset data
x=tf.placeholder("float",[None, inputN])
y=tf.placeholder("float",[None, outputN])

def NN(x,w,b):
    
    HL1 = tf.add(tf.matmul(x, w['h1']), b['b1']) 
    HL1 = tf.nn.sigmoid(HL1)                               # sigmoid(HL1)
#    # Hidden layer 2
    HL2 = tf.add(tf.matmul(HL1, w['h2']), b['b2']) 
    HL2 = tf.nn.sigmoid(HL2)                             # sigmoid(HL)
    
    # Output layer
    out_layer = tf.matmul(HL2, w['out']) + b['out'] 

    
    return out_layer
    
def acuracytestNN():
    # Test model
    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    return accuracy.eval({x: testx, y: testy})

weights = {
    'h1': tf.Variable(tf.random_normal([inputN, hiddenN[0]])),  
    'h2': tf.Variable(tf.random_normal([hiddenN[0], hiddenN[1]])),
    'out': tf.Variable(tf.random_normal([hiddenN[1], outputN]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([hiddenN[0]])),
    'b2': tf.Variable(tf.random_normal([hiddenN[1]])),
    'out': tf.Variable(tf.random_normal([outputN]))
}

predict = NN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, 
                                                                     labels=y))
#AdamOptimizer works better than GradientDescentOpt.
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)


init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(trainingEpochs):
        #avg_cost = 0.
        #total_batch = int(len(trainx)/batchSize)
#        # Loop over all batches
#        for i in range(total_batch):
#            a=list(range(batchSize))
#            batch_x= trainx[a]
#            batch_y= trainy[a]

        _, cNN = sess.run([optimizer, cost], feed_dict={x: trainx,
                                                      y: trainy})
        t=t+1
        # Compute average loss
        #avg_cost += cNN / total_batch
#            avg_costPL += cPL / total_batch

        if t % 10 == 0:
            
            acunn=acuracytestNN()
#            acupl=acuracytestPL()
            ttt= np.append(ttt,t)
            nnnnnn= np.append(nnnnnn,acunn)
#            plplpl= np.append(plplpl,acupl)
            print('t=',t,acunn)
        plt.plot(ttt,nnnnnn,'r--')
        plt.show()
        
    print("Optimization Finished!")
    print ("Neural Network accuracy:",acuracytestNN())
#    print ("+PL:",acuracytestPL())

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("", one_hot=True)


#Neural Network parameters
ttt=np.zeros(1)
nnnnnn=np.zeros(1)
plplpl=np.zeros(1)
acunn=0
acupl=0
learningRate=0.001
trainingEpochs=20
batchSize=200
inputN=784
hiddenN=[20,18,16,14,12]
outputN=10
PLbatchSize=256
t=0
cPL=0;
T1=1000
T2=6000
a=0.
af=3.

print("HiddenLayer1:",hiddenN[0],"HiddenLayer2:",hiddenN[1])
x=tf.placeholder("float",[None, inputN])
y=tf.placeholder("float",[None, outputN])
PLx=tf.placeholder("float",[None, inputN])
PLy=tf.placeholder("float",[None, outputN])
alpha=tf.placeholder("float",)
plt.clf()


def NN(x,w,b):
    
#    HLn=np.shape(hiddenN)[0]
#    print("Number of hidden layers:",HLn)
#    HL=tf.Variable([[(1,tf.zeros(hiddenN[0])],[tf.zeros(hiddenN[1])]])
#    print(HL[1])
#    HL[0]=tf.add(tf.matmul(x, w['h1']), b['b1'])
#    HL[0]=tf.nn.sigmoid(HL[0])
#    HL[1]=tf.add(tf.matmul(HL[0], w['h1']), b['b1'])
#    HL[1]=tf.nn.sigmoid(HL[1])
#    # Hidden layer 1
    HL = tf.add(tf.matmul(x, w['h1']), b['b1']) 
    HL = tf.nn.sigmoid(HL)                               # sigmoid(HL)
#    # Hidden layer 2
    HL2 = tf.add(tf.matmul(HL, w['h2']), b['b2']) 
    HL2 = tf.nn.sigmoid(HL2)                             # sigmoid(HL)
    
#    # Hidden layer 3
    HL3 = tf.add(tf.matmul(HL2, w['h3']), b['b3']) 
    HL3 = tf.nn.sigmoid(HL3)                             # sigmoid(HL)
    
#    # Hidden layer 4
    HL4 = tf.add(tf.matmul(HL3, w['h4']), b['b4']) 
    HL4 = tf.nn.sigmoid(HL4)                             # sigmoid(HL)

#    # Hidden layer 5
    HL5 = tf.add(tf.matmul(HL4, w['h5']), b['b5']) 
    HL5 = tf.nn.sigmoid(HL5)                             # sigmoid(HL)
    

    
    # Output layer
    out_layer = tf.matmul(HL5, w['out']) + b['out'] 

    
    return out_layer

#initialize weights and biases

weightsNN = {
    'h1': tf.Variable(tf.random_normal([inputN, hiddenN[0]])),    #784x256
    'h2': tf.Variable(tf.random_normal([hiddenN[0], hiddenN[1]])),
    'h3': tf.Variable(tf.random_normal([hiddenN[1], hiddenN[2]])),    #784x256
    'h4': tf.Variable(tf.random_normal([hiddenN[2], hiddenN[3]])),
    'h5': tf.Variable(tf.random_normal([hiddenN[3], hiddenN[4]])),
    'out': tf.Variable(tf.random_normal([hiddenN[4], outputN]))  #256x10
}
biasesNN = {
    'b1': tf.Variable(tf.random_normal([hiddenN[0]])),             #256x1
    'b2': tf.Variable(tf.random_normal([hiddenN[1]])),
    'b3': tf.Variable(tf.random_normal([hiddenN[2]])),             #256x1
    'b4': tf.Variable(tf.random_normal([hiddenN[3]])),
    'b5': tf.Variable(tf.random_normal([hiddenN[4]])),
    'out': tf.Variable(tf.random_normal([outputN]))              #10x1
}
weightsPL = {
    'h1': tf.Variable(tf.random_normal([inputN, hiddenN[0]])),    #784x256
    'h2': tf.Variable(tf.random_normal([hiddenN[0], hiddenN[1]])),
    'h3': tf.Variable(tf.random_normal([hiddenN[1], hiddenN[2]])),    #784x256
    'h4': tf.Variable(tf.random_normal([hiddenN[2], hiddenN[3]])),
    'h5': tf.Variable(tf.random_normal([hiddenN[3], hiddenN[4]])),
    'out': tf.Variable(tf.random_normal([hiddenN[4], outputN]))  #256x10
}
biasesPL = {
    'b1': tf.Variable(tf.random_normal([hiddenN[0]])),             #256x1
    'b2': tf.Variable(tf.random_normal([hiddenN[1]])),
    'b3': tf.Variable(tf.random_normal([hiddenN[2]])),             #256x1
    'b4': tf.Variable(tf.random_normal([hiddenN[3]])),
    'b5': tf.Variable(tf.random_normal([hiddenN[4]])),
    'out': tf.Variable(tf.random_normal([outputN]))              #10x1
}


predNN = NN(x, weightsNN, biasesNN)
predPL = NN(x, weightsPL, biasesPL)
predPL1 = NN(PLx, weightsPL, biasesPL)


costNN = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predNN, 
                                                                     labels=y))

costPL = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predPL, 
                                                                     labels=y)),
              (alpha*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predPL1,
                                                                                       labels=PLy))))

# Gradient Descent
#optimizerNN = tf.train.GradientDescentOptimizer(learningRate).minimize(costNN)
optimizerNN = tf.train.AdamOptimizer(learningRate).minimize(costNN)
optimizerPL = tf.train.AdamOptimizer(learningRate).minimize(costPL)
# Initializing the variable
init = tf.global_variables_initializer()

# Launch the graph
def acuracytestNN():
    # Test model
    correct_prediction = tf.equal(tf.argmax(predNN, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    return accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    
def acuracytestPL():
    # Test model
    correct_prediction = tf.equal(tf.argmax(predPL, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    return accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(trainingEpochs):
        avg_costNN = 0.
        avg_costPL = 0.
        total_batch = int(60000/batchSize)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batchSize)

            _, cNN = sess.run([optimizerNN, costNN], feed_dict={x: batch_x,
                                                          y: batch_y})

                    
            #Pseudolabel
#            if t>T1:
#                a=((t-T1)/(T2-T1))*af
#                if t>T2:
#                    a=af
#            batch_xpred,yy= mnist.train.next_batch(PLbatchSize)
#            batch_ypred = sess.run([predPL], feed_dict={x: batch_xpred})
#            batch_ypred=batch_ypred[0]
#            batch_ypred=batch_ypred.argmax(1)
#            kk=np.zeros((PLbatchSize,10))
#            for ii in range(PLbatchSize):
#                kk[ii,batch_ypred[ii]]=1
#            
#            _,cPL = sess.run([optimizerPL, costPL], feed_dict={x: batch_x,
#                                                          y: batch_y,
#                                                          PLx: batch_xpred,
#                                                          PLy: kk,
#                                                          alpha: a})
            t=t+1
            # Compute average loss
            avg_costNN += cNN / total_batch
#            avg_costPL += cPL / total_batch

        if t % 100 == 0:
            print('t=',t)
            acunn=acuracytestNN()
#            acupl=acuracytestPL()
            ttt= np.append(ttt,t)
            nnnnnn= np.append(nnnnnn,acunn)
#            plplpl= np.append(plplpl,acupl)
        plt.plot(ttt,nnnnnn,'r--')
        plt.show()
        
    print("Optimization Finished!")
    print ("Neural Network accuracy:",acuracytestNN())
#    print ("+PL:",acuracytestPL())
"""