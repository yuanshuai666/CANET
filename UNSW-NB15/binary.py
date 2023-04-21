import pandas as pd
import numpy as np
from keras.layers import LSTM, SimpleRNN, GRU, Bidirectional, BatchNormalization,Convolution1D,MaxPooling1D, Reshape, GlobalAveragePooling1D
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
# from imblearn.over_sampling import RandomOverSampler
from keras.layers import Dense, Embedding, Flatten, Input, Attention, Conv1D, Concatenate
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import confusion_matrix
import os
from functools import partial
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()

df = pd.read_csv('/data6/shuaiYuan/dataset/UNSW-NB15 - CSV Files/UNSW-NB15 - CSV Files/a part of training and testing set/UNSW_NB15_testing-set.csv')
qp = pd.read_csv('/data6/shuaiYuan/dataset/UNSW-NB15 - CSV Files/UNSW-NB15 - CSV Files/a part of training and testing set/UNSW_NB15_training-set.csv')
# print(qp.head())
# Dropping the last columns of training set
df = df.drop('id', 1)
df = df.drop('attack_cat', 1)
qp = qp.drop('id', 1)
qp = qp.drop('attack_cat', 1)
cols = ['proto','state','service']
# print(df.head())

# One-hot encoding
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each, 1)
    return df


combined_data = pd.concat([df,qp])
tmp = combined_data.pop('label')

combined_data = one_hot(combined_data,cols)


def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy() # do not touch the original df
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


new_train_df = normalize(combined_data,combined_data.columns)
# new_train_df = combined_data
new_train_df["Class"] = tmp
y_train = new_train_df["Class"]
# print(new_train_df)
combined_data_X = new_train_df.drop('Class', 1)
oos_pred = []
DR = []
FPR = []
F1 = []

kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

@tf.keras.utils.register_keras_serializable()
class ScaledDotProductAttention(tf.keras.layers.Layer):

    def __init__(self, dropout_rate=0., **kwargs):
        self._dropout_rate = dropout_rate
        self._masking_num = -2**32+1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)
        outputs = inputs + masks * self._masking_num
        return outputs

    def future_mask(self, inputs):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * self._masking_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        return outputs

    def call(self, inputs):
        queries, keys, values = inputs
        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')

        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1])) # MatMul
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale

        softmax_out = K.softmax(scaled_matmul) # SoftMax
        # Dropout
        out = K.dropout(softmax_out, self._dropout_rate)

        outputs = K.batch_dot(out, values)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, n_heads, head_dim=9, dropout_rate=.1, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):

        queries, keys, values = inputs
        queries_linear = K.dot(queries, self._weights_queries)
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)

        att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]

        attention = ScaledDotProductAttention(dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)

        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

class EQLv2(tf.keras.losses.Loss):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=2,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False, **kwargs):

        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

        def _func(x, gamma, mu):
            return 1 / (1 + tf.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
        super(EQLv2, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        label, cls_score = y_true, y_pred
        self.n_i, self.n_c = cls_score.shape[0], cls_score.shape[1]     # none 10
        self.gt_classes = label
        self.pred_class_logits = cls_score
        target = tf.cast(label, dtype='float32')
        pos_w, neg_w = self.get_weight(cls_score)
        print(pos_w)
        print(target)
        weight = pos_w * target + neg_w * (1 - target)  # (None, 10)
        cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(target, cls_score)
        cls_loss = tf.reduce_sum(cls_loss * weight) / self.n_c

        # self.collect_grad(cls_score.detach(), target.detach(), weight.detach())
        self.collect_grad(cls_score, target, weight)
        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = tf.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = tf.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = tf.abs(grad)
        # print("&&&&&&&&", grad)
        # print(tf.reduce_sum(grad * target * weight))
        # do not collect grad for objectiveness branch [:-1]
        # pos_grad = tf.reduce_sum(grad * target * weight)[:-1]
        pos_grad = tf.reduce_sum(grad * target * weight)
        neg_grad = tf.reduce_sum(grad * (1 - target) * weight)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, cls_score):
        # we do not have information about pos grad and neg grad at beginning
        if self._pos_grad is None:
            self._pos_grad = tf.zeros(self.num_classes)
            self._neg_grad = tf.zeros(self.num_classes)
            neg_w = tf.ones(self.n_c)
            pos_w = tf.ones(self.n_c)
        else:
            # the negative weight for objectiveness is always 1
            # neg_w = tf.concat([self.map_func(self.pos_neg), tf.ones(1)], axis=0)
            neg_w = self.map_func(self.pos_neg)
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = tf.reshape(neg_w,shape=(1,-1))
            pos_w = tf.reshape(pos_w,shape=(1,-1))
            # pos_w = pos_w.view(1, -1)
        return pos_w, neg_w


def build_model():

    input = Input(shape=(196, 1), dtype='float32')
    cnn_layer = Conv1D(filters=64, kernel_size=64, strides=1, padding='same',activation='relu')(input)
    pool = MaxPooling1D(pool_size=4)(cnn_layer)
    norm = BatchNormalization()(pool)
    attention = Attention()([norm, norm])
    cnn_layer2 = Conv1D(filters=128, kernel_size=64, strides=1, padding='same',activation='relu')(attention)
    pool2 = MaxPooling1D(pool_size=2)(cnn_layer2)
    norm2 = BatchNormalization()(pool2)
    attention2 = Attention()([norm2, norm2])
    cnn_layer3 = Conv1D(filters=256, kernel_size=64, strides=1, padding='same',activation='relu')(attention2)
    pool3 = MaxPooling1D(pool_size=2)(cnn_layer3)
    norm3 = BatchNormalization()(pool3)
    attention3 = Attention()([norm3, norm3])
    flatten = Flatten()(attention3)
    output = Dense(2)(flatten)
    model = Model(inputs=input, outputs=output)
    eqloss = EQLv2()
    model.compile(optimizer='adam', loss=eqloss, metrics=['accuracy'])
    model.summary()
    return model

model = build_model()
for train_index, test_index in kfold.split(combined_data_X,y_train):
    train_X, test_X = combined_data_X.iloc[train_index], combined_data_X.iloc[test_index]
    train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]

    # print(train_y.value_counts())
    train_X_over,train_y_over = train_X, train_y

    x_columns_train = new_train_df.columns.drop('Class')
    x_train_array = train_X_over[x_columns_train].values
    x_train_1=np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))

    dummies = pd.get_dummies(train_y_over) # Classification
    outcomes = dummies.columns
    num_classes = len(outcomes)
    y_train_1 = dummies.values

    x_columns_test = new_train_df.columns.drop('Class')
    x_test_array = test_X[x_columns_test].values
    x_test_2=np.reshape(x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1))

    dummies_test = pd.get_dummies(test_y) # Classification
    outcomes_test = dummies_test.columns
    num_classes = len(outcomes_test)
    y_test_2 = dummies_test.values

    # print(y_train_1)
    history = model.fit(x_train_1, y_train_1,validation_data=(x_test_2,y_test_2), epochs=100, batch_size=256)

    pred = model.predict(x_test_2)

    pred = np.argmax(pred,axis=1)
    y_eval = np.argmax(y_test_2,axis=1)
    score = metrics.accuracy_score(y_eval, pred)
    recall = metrics.recall_score(y_eval, pred)
    cm = metrics.confusion_matrix(y_eval, pred)
    f1 = metrics.f1_score(y_eval, pred)
    fpr = cm[1][0]/(cm[1][0]+cm[1][1])
    oos_pred.append(score)
    DR.append(recall)
    FPR.append(fpr)
    F1.append(f1)
    print(cm)
    print("ACC: {}".format(score))
    print("DR: ", recall)
    print("FPR: ", fpr)
    print("F1: ", f1)


print(oos_pred)
print(DR)
print(FPR)
print(F1)