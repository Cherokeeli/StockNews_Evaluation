import numpy as np
import tensorflow as tf
# '\u4E00'19968 && c<='\u9FA5‘ 40869


def convertToMatrix(words):
    word_matrix = np.zeros((104, 104))
    if (words ==''):
        return word_matrix
    x = 0
    y = 0
    total = 0
    prex = 0
    prey = 1
    for word in words:
        total += (ord(word) - 19968)
    bias = total / (3 * 40869)

    for i in range(len(words)):
        code = ord(words[i])
        x = int((code - 19968) / 2 / 104)
        print('x: ', y + int(bias * (104 - y)) + 1)
        y = int((code - 19968) / 2) % 104
        if (prex / prey > 1):
            word_matrix[x + int(bias * (104 - x)) + 1][y +
                                                   int(bias * (104 - y)) + 1] = 1
        else:
            word_matrix[x + int(bias * (104 - x)) - 1][y +
                                                   int(bias * (104 - y)) - 1] = 1
        prex, prey = x, y
    # words[i]-19968
    return word_matrix
a = np.array([])
b = np.array([3,2,1])
c = np.array([1,2,3])
d = np.append(b,c)
e = d.reshape(2,3)
print(e)
# print(np.concatenate((e,c)))

a = tf.constant([1,2,3,4,5,6],shape=[1,6])
b = tf.constant([1,2,3,4,5,6,1,2,3,4,5,6], shape=[6,2])
c = tf.matmul(a,b)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print(sess.run(c))
#print(a)