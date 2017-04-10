import tensorflow as tf
import numpy as np
import pandas as pd
import math

# Data preloded
def convertToMatrix(words):
    word_matrix = np.zeros((104, 104))
    if (words):
        return word_matrix
    x = 0
    y = 0
    total = 0
    prex = 0
    prey = 1
    length = (len(words))
    # print('len of words: ', words)
    for i in range(length):
        # print('words: ',words)
        total = total+(int(ord(words[i])) - 19968)
        # print('each total: ',total)
    bias = total / (len(words) * 21000)
    # print('bias: ',bias)

    for i in range(len(words)):
        code = ord(words[i])
        # print('code: ',code)
        x = int((code - 19968) / 2 / 104)
        # print('y: ', y + int(bias * (104 - y)) + 1)
        y = int((code - 19968) / 2) % 104
        # print('x: ',x + int(bias * (104 - x)))
        # print('y: ',y + int(bias * (104 - y)))
        if (prex / prey > 1):
            word_matrix[x + int(bias * (104 - x)) + 1][y +
                                                   int(bias * (104 - y)) + 1] = 1
        else:
            word_matrix[x + int(bias * (104 - x)) - 1][y +
                                                   int(bias * (104 - y)) - 1] = 1
        prex, prey = x, y
    # words[i]-19968
    return word_matrix

def dataPreloaded(path):
    dataset = pd.read_csv(path)
    y_increase = np.array(dataset['increase'],dtype=int)
    y_decrease = np.array([v^1 for v in y_increase])
    x_author = np.array(dataset['author'],dtype=str)
    authors = np.array([])
    for author in x_author:
        temp = convertToMatrix(author)
        print(len(x_author))
        authors = np.append(authors, temp)
    authors = authors.reshape(len(dataset), 10816)
    train_y = np.column_stack((y_increase, y_decrease))
    train_x = np.array(dataset.iloc[:len(dataset),0:45])
    return train_x, train_y, authors
# for v in y_set:
#     batch = np.array([v, (v^1)]) # generate reverse label array
#     # print(batch)
#     train_y = np.vstack((train_y, batch))
x_train, y_train, a_train = dataPreloaded('trend_train-author.csv')
x_test, y_test, a_test = dataPreloaded('trend_test-author.csv')

print(x_train.shape,y_train.shape,a_train.shape,x_test.shape,y_test.shape,a_test.shape)


# print("shape is ",train_x)
# Model parameter
W = tf.Variable(tf.zeros([45, 2]))
b = tf.Variable(tf.zeros([2]))

W1 = tf.Variable(tf.zeros([10816, 2]))
b1 = tf.Variable(tf.zeros([2]))
# Model input and output
x = tf.placeholder(tf.float32,shape=(None,45))
x1 = tf.placeholder(tf.float32, shape=(None, 10816))
y = tf.nn.softmax((tf.matmul(x, W)) + (tf.matmul(x1, W1) + b1))
y_ = tf.placeholder(tf.float32, shape=(None,2))

# Loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Optimizer
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

def next_batch(train, test, author, i, size):
    return train[i*size:(i+1)*size], test[i*size:(i+1)*size], author[i*size:(i+1)*size]

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# print(train_x.shape)
for i in range(500):
    # print("x: ",np.array([1,2,3,4]).shape)
    if (i%5==0):
        print('train: %d times, total: %d' % (i/5,100))
    batch_x, batch_y, batch_a = next_batch(x_train, y_train, a_train,i,100)
    # print(batch_y)
    sess.run(train_step, feed_dict={x: batch_x, x1:batch_a, y_: batch_y})

correct_prediction = tf.equal(tf.argmax(y,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: x_test, x1:a_test, y_: y_test}))
