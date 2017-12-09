"""
Artificial Intelligence & Robotics
Biometric systems
Palmvein recognition / Deep Neural Networks
Jose Vicente Jaramillo

*** Objetive ***


*** Results ***

subject0: 0.975 accu after 500 1000 1500 epochs
1: 0.98
2:


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
print(len(testx[1]))
inputN=len(testx[1])
hiddenN=[204,100]
outputN=2
trainingEpochs=1000
learningRate=0.001
batchSize=400
t=0
ttt=np.zeros(1)
nnnnnn=np.zeros(1)
trainingsubject=0   # 0-49

#Set output as a binary yes/no 
#trainys=list(range(len(trainy)))

trainxx=np.asarray(trainx)
testxx=np.asarray(testx)
testys=np.zeros((len(testy),2))
trainys=np.zeros((len(trainy),2))

for i in range(len(trainy)):
    trainys[i,0]= trainy[i][trainingsubject]
    if(trainys[i,0]==0.0):
        trainys[i,1]= 1.0
    else:
        trainys[i,1]= 0.0
trainys=np.asarray(trainys)

for i in range(len(testy)):
    testys[i,0]= testy[i][trainingsubject]
    if(testys[i,0]==0.0):
        testys[i,1]= 1.0
    else:
        testys[i,1]= 0.0
testys=np.asarray(testys)

#Populate the training data with subject data to avoid unbalanced data
p=np.argwhere(trainys[:,0])
pp=trainxx[p[:,0]]
pa=trainys[p[:,0]]
pp=np.tile(pp,(50,1))
pa=np.tile(pa,(50,1))
trainxx=np.append(trainxx,pp,axis=0)
trainys=np.append(trainys,pa,axis=0)

print("HiddenLayer1:",hiddenN[0],"HiddenLayer2:",hiddenN[1])

#Tensors reserved for the dataset data
x=tf.placeholder("float",[None, inputN])
y=tf.placeholder("float",[None, outputN])

def NN(x,w,b):
    
    HL1 = tf.add(tf.matmul(x, w['h1']), b['b1']) 
    HL1 = tf.nn.relu(HL1)                               # sigmoid(HL1)
#    # Hidden layer 2
    HL2 = tf.add(tf.matmul(HL1, w['h2']), b['b2']) 
    HL2 = tf.nn.relu(HL2)                             # sigmoid(HL)
    
    # Output layer
    out_layer = tf.matmul(HL2, w['out']) + b['out'] 

    
    return out_layer
    
def acuracytestNN():
    # Test model
    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    return accuracy.eval({x: testxx, y: testys})

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

        #_, cNN = sess.run([optimizer, cost], feed_dict={x: trainxx,
        #                                              y: trainys})
        
#        NNNN=sess.run([predict],feed_dict={x:testxx})
#        print(NNNN)
        _, cNN = sess.run([optimizer, cost], feed_dict={x: trainxx,
                                                      y: trainys})
        t=t+1
        # Compute average loss
        print(cNN)
#            avg_costPL += cPL / total_batch

        if t % 10 == 0:
            NNNN=sess.run([predict],feed_dict={x:testxx})
            print(NNNN)
            NNNN=sess.run([predict],feed_dict={x:trainxx})
            print(NNNN)
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