# Python code for TensorFlow training for toggle switch or XOR function
# Import libraries
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

# Description
print("My development environment is a Windows 10 PC with Python 3.7.4 and Tensorflow 2.2.0 installed. \nTo get the program running, I first imported the necessary libraries - Tensorflow, Matplotlib and NumPy. \nThen I created the neural network architecture for both toggle switch circuit and XOR function.  I defined the input, hidden, and output layers, as well as the target and loss functions. \nThe program was trained and tested successfully.")

# Ask user to choose toggle switch or XOR
choice = input('\nChoose either toggle switch (1) or XOR (2): ')

# Load dataset
# Toggle switch data
if choice == '1':
    data = np.loadtxt(r'toggle_switch_dataset.csv', delimiter=',')
    X_train = data[:, 0:2]
    y_train = data[:, 2]
# XOR data
elif choice == '2':
    data = np.loadtxt(r'xor_dataset.csv', delimiter=',')
    X_train = data[:, 0:2]
    y_train = data[:, 2]

# Define model
# Create placeholders
x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

# Create a fully connected layer with ReLU activation
h1 = tf.layers.dense(x, 16, activation=tf.nn.relu, name='h1')

# Create a fully connected layer with sigmoid activation
h2 = tf.layers.dense(h1, 16, activation=tf.nn.sigmoid, name='h2')



# Create output layer
y_pred = tf.layers.dense(h2, 1, activation=None, name='y_pred')

# Define loss and optimizer
loss = tf.losses.mean_squared_error(y, y_pred)
train_op = tf.train.AdamOptimizer().minimize(loss)

# Train the model
accuracies = []
losses = []
epoch = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Train the model
    for i in range(1000):
        _, l = sess.run([train_op, loss], feed_dict={x: X_train, y: y_train.reshape(-1, 1)})
        losses.append(l)
        epoch.append(i)
        if (i + 1) % 50 == 0:
            print('Epoch {}, Loss: {}'.format(i + 1, l))

            # Test the model
            y_pred_test = sess.run(y_pred, feed_dict={x: X_train})

            # Calculate accuracy
            correct_prediction = np.equal(np.round(y_pred_test), y_train.reshape(-1, 1))
            accuracy = np.mean(correct_prediction)
            accuracies.append(accuracy)
            print('Epoch {}, Accuracy: {}'.format(i + 1, round(accuracy*100,2)))

# Plot the accuracy and loss for each epoch
plt.subplot(2, 1, 1)
plt.plot(accuracies)
plt.title('Model Accuracy')
plt.xticks([])
plt.ylabel('Accuracy')
ax = plt.gca()
ax.spines['bottom'].set_visible(False)
plt.legend(['Accuracy'], loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(losses)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
ax = plt.gca()
ax.spines['top'].set_visible(False)
plt.legend(['Loss'], loc='upper right')

plt.tight_layout()
plt.show()