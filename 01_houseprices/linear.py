import tensorflow as tf
import numpy as np
import pandas
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 2000

# load data
data = pandas.read_csv('data.csv')
data = data[['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt', 'SalePrice']]

# Convert data
#data['SalePrice'] = np.log(data['SalePrice'])

# Split train and test data
train = data.sample(frac=0.8, random_state=200)
test = data.drop(train.index)

# Replace null data with 0
train_X, train_Y = train[['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt']].fillna(0), train[['SalePrice']]
test_X, test_Y = test[['GrLivArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt']].fillna(0), test[['SalePrice']]

num_features = train_X.shape[1]

# Create variables
# X is a symbolic variable which will contain input data
# shape [None, num_features] suggests that we don't limit the number of instances in the model
# while the number of features is known in advance
X = tf.placeholder("float", [None, num_features])
# same with labels: number of classes is known, while number of instances is left undefined
Y = tf.placeholder("float", [None, 1])

# W - weights array
W = tf.Variable(tf.zeros([num_features, 1]))
# B - bias array
B = tf.Variable(tf.zeros([1]))

# Define a model
# a simple linear model y=wx+b
Y_model = tf.matmul(X, W) + B
# Y_model will contain predictions the model makes, while Y contains real data

# Define a cost function
cost = tf.reduce_sum(tf.square(Y - Y_model))

# Define an optimizer
# I prefer Adam
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# but there is also a plain old SGD if you'd like
#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Error calculation
total_error = tf.reduce_sum(tf.square(Y - tf.reduce_mean(Y)))
unexplained_error = tf.reduce_sum(tf.square(Y - Y_model))
R_squared = 1.0 - tf.div(unexplained_error, total_error)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_err = 0
    for epoch in range(training_epochs):
        # run an optimization step with all train data
        err, _ = sess.run([cost, train_op], feed_dict={X: train_X, Y: train_Y})
        print(epoch, err)

    w_val = sess.run(W, feed_dict={X: train_X, Y: train_Y})

    print w_val

    print('R squared Train', sess.run(R_squared, feed_dict={X: train_X, Y: train_Y}))
    print('R squared Test', sess.run(R_squared, feed_dict={X: test_X, Y: test_Y}))

    y_predict = sess.run(Y_model, feed_dict={X: test_X, Y: test_Y})
    plt.scatter(test_Y, y_predict)
    plt.show()


