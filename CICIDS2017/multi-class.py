import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Input, Attention, Conv1D, BatchNormalization, MaxPooling1D
from keras.models import Model
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn import metrics
from functools import partial
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()
data = np.load('/data6/shuaiYuan/code/Keras_bearing_fault_diagnosis-master/CICIDS2017/data/data.npy')
label = np.load('/data6/shuaiYuan/code/Keras_bearing_fault_diagnosis-master/CICIDS2017/data/label.npy', allow_pickle=True)

x = np.expand_dims(data, 2)
n_obs, feature, depth = x.shape
y = label.squeeze()
y = y.astype('int64')
# print(type_of_target(y))
acc = []
dr = []
fpr = []
F1 = []


class EQLv2(tf.keras.losses.Loss):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=15,  # 1203 for lvis v1.0, 1230 for lvis v0.5
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
        target = label
        pos_w, neg_w = self.get_weight(cls_score)

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
            neg_w = self.map_func(self.pos_neg)
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = tf.reshape(neg_w,shape=(1,-1))
            pos_w = tf.reshape(pos_w,shape=(1,-1))
        return pos_w, neg_w

def build_model():

    input = Input(shape=(feature, depth), dtype='float32')
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
    output = Dense(15)(flatten)
    model = Model(inputs=input, outputs=output)
    eqloss = EQLv2()
    model.compile(optimizer='adam', loss=eqloss, metrics=['accuracy'])
    model.summary()
    return model


model = build_model()

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(x, y):
    train_X, test_X = x[train_index], x[test_index]
    train_y, test_y = y[train_index], y[test_index]
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    history = model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=10, batch_size=256)

    pred = model.predict(test_X)
    pred = np.argmax(pred, axis=1)
    y_eval = np.argmax(test_y, axis=1)
    score = metrics.accuracy_score(y_eval, pred)
    acc.append(score)
    target_names = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye', 'FTP-Patator', 'SSH-Patator', 'DoS slowloris',
                    'DoS Slowhttptest', 'Bot', 'Web Attack Brute Force', 'Web Attack XSS', 'Infiltration',
                    'Web Attack Sql Injection', 'Heartbleed']
    print(classification_report(y_eval, pred, target_names=target_names))
    cm = metrics.confusion_matrix(y_eval, pred)
    print(cm)
    tp = cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5]+cm[6][6]+cm[7][7]+cm[8][8]+cm[9][9]+cm[10][10]+cm[11][11]+cm[12][12]+cm[13][13]+cm[14][14]
    fn = cm[1][0]+cm[2][0]+cm[3][0]+cm[4][0]+cm[5][0]+cm[6][0]+cm[7][0]+cm[8][0]+cm[9][0]+cm[10][0]+cm[11][0]+cm[12][0]+cm[13][0]+cm[14][0]
    fp = cm[0][1]+cm[0][2]+cm[0][3]+cm[0][4]+cm[0][5]+cm[0][6]+cm[0][7]+cm[0][8]+cm[0][9]+cm[0][10]+cm[0][11]+cm[0][12]+cm[0][13]+cm[0][14]
    tn = cm[0][0]
    DR = tp/(tp+fn)
    FPR = fp/(fp+tn)
    f1 = metrics.f1_score(y_eval, pred, average='weighted')
    dr.append(DR)
    fpr.append(FPR)
    F1.append(f1)
    print(cm)
    print("ACC: {}".format(score))
    print("DR=recall:", DR)
    print("FPR:", FPR)
    print("f1:", f1)

print(acc)
print(dr)
print(fpr)
print(F1)

