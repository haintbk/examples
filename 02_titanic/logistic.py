import tensorflow as tf
import pandas

learning_rate = 0.01
training_epochs = 2000

# load data
data = pandas.read_csv('data.csv')
data = data[['Age', 'Sex', 'SibSp', 'Fare', 'Survived']]

# Convert data
data['Sex'] = data['Sex'].astype('category').cat.codes

# Split train and test data
train = data.sample(frac=0.8, random_state=200)
test = data.drop(train.index)

# Replace null data with 0
train_X, train_Y = train[['Age', 'Sex', 'SibSp', 'Fare']].fillna(0), train[['Survived']]
test_X, test_Y = test[['Age', 'Sex', 'SibSp', 'Fare']].fillna(0), test[['Survived']]

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
# a simple linear model y=wx+b wrapped into sigmoid
Y_model = tf.nn.sigmoid(tf.matmul(X, W) + B)
# Y_model will contain predictions the model makes, while Y contains real data

# Define a cost function
cost = tf.reduce_mean(-tf.log(Y_model * Y + (1 - Y_model) * (1 - Y)))

# Define an optimizer
# I prefer Adam
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# but there is also a plain old SGD if you'd like
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Define accuracy calculation
correct_prediction = tf.equal(Y, tf.to_float(tf.greater(Y_model, 0.5)))
accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_err = 0
    for epoch in range(training_epochs):
        # run an optimization step with all train data
        err, _ = sess.run([cost, train_op], feed_dict={X: train_X, Y: train_Y})
        print(epoch, err)

    w_val = sess.run(W, feed_dict={X: train_X, Y: train_Y})

    print('accuracy', sess.run(accuracy, feed_dict={X: test_X, Y: test_Y}))

