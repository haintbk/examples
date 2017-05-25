import pandas

import tensorflow as tf

# load data
data = pandas.read_csv("data.csv")
data = data[['Age', 'Sex', 'SibSp', 'Fare', 'Survived']].fillna(0)

# Split train and test data
train = data.sample(frac=0.8, random_state=200)
test = data.drop(train.index)

CONTINUOUS_COLUMNS = ['Age', 'SibSp', 'Fare']
CATEGORICAL_COLUMNS = ['Sex']
LABEL_COLUMN = 'Survived'


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.

    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}

    # Merges the two dictionaries into one.
    # For python3
    #feature_cols = {**continuous_cols, **categorical_cols}

    # For python2
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())

    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)

    # Returns the feature columns and the label.
    return feature_cols, label


def train_input_fn():
    return input_fn(train)


def eval_input_fn():
    return input_fn(test)


# replace null data with 0
train_X, train_Y = train[['Age', 'Sex', 'SibSp', 'Fare']].fillna(0), train[['Survived']]
test_X, test_Y = test[['Age', 'Sex', 'SibSp', 'Fare']].fillna(0), test[['Survived']]

# Create variables
X_age = tf.contrib.layers.real_valued_column('Age')
X_sex = gender = tf.contrib.layers.sparse_column_with_keys(column_name="Sex", keys=["Female", "Male"])
X_sibsp = tf.contrib.layers.real_valued_column('SibSp')
X_fare = tf.contrib.layers.real_valued_column('Fare')

m = tf.contrib.learn.LinearClassifier(feature_columns=[X_age, X_sex, X_sibsp, X_fare])
m.fit(input_fn=train_input_fn, steps=2000)

results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))



