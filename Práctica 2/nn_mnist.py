import gzip
import pickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
train_set, valid_set, test_set = u.load()
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set;

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print (train_y[57])


# TODO: the neural net!!



train_y = one_hot(train_y, 10)  # the labels are in the last row. Then we encode them in one hot code
testeo_y = one_hot(test_y, 10)  # the labels are in the last row. Then we encode them in one hot code



x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

#W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1) #capa
#b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

#W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1) #salida de las neuronas
#b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)


W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1) #capa
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1) #salida de las neuronas
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)


h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01 <-- la mejor
# train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # learning rate: 0.1
# train = tf.train.GradientDescentOptimizer(0.005).minimize(loss)  # learning rate: 0.005
# train = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)  # learning rate: 0.0005 <--raro

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20

training_errors = []
validation_errors = []
epoch = 0
sta = 0
while sta < 20:
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # gráfica de evolución del entrenamiento.
    training_error = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}) / batch_size

    # gráfica de evolución de la validación.
    validation_error = sess.run(loss, feed_dict={x: valid_x, y_:one_hot(valid_y, 10)})/ len(testeo_y)

    # añadir los valores a la lista
    validation_errors.append(validation_error)
    training_errors.append(training_error)
    print("Epoch #:", epoch, "Error: ", training_error)
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print(b, "-->", r)
    print("----------------------------------------------------------------------------------")
    print ("Epoca: ", epoch)
    epoch = epoch + 1
    print("Error de entrenamiento: ", training_error)
    print("Error de validacion: ", validation_error)
    print("Diferencia de error de validación: ", abs(float (validation_errors[epoch-2]) - float(validation_errors[epoch-1])))
    print("Estabilización: ", sta)
    if (abs(validation_errors[epoch-1] - validation_errors[epoch-2]) < 0.01):
       sta+=1
    else:
        sta = 0
# Test
print ("Resultados del test:")
print ("----------------------------------------------------------------------------------")
result = sess.run(y, feed_dict={x: test_x})

success = 0
fail = 0
for b, r in zip(testeo_y, result):
    if (np.argmax(b) != np.argmax(r)):
        fail += 1

print ("Numero de aciertos: " , len(test_x) - fail)
print ("Numero de fallos: " , fail)
print ("Numero total: " , len(test_x))
print ("Porcentaje de aciertos: " , (float(len(test_x) - fail) / float(len(test_x))) * 100 , "%")
print ("----------------------------------------------------------------------------------")

plt.plot(validation_errors)
plt.plot(training_errors)
plt.legend(['Error de validacion', 'Error de Entrenamiento'])
plt.show()
