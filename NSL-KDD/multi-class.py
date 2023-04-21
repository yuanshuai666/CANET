import pandas as pd
import numpy as np
from keras.layers import BatchNormalization, MaxPooling1D
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Flatten, Input, Attention, Conv1D
from keras.models import Model
import tensorflow as tf
from sklearn.metrics import classification_report
import os
from functools import partial
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()
df = pd.read_csv('/data6/shuaiYuan/dataset/NSL-KDD/KDDTrain+.txt', header=None)
qp = pd.read_csv('/data6/shuaiYuan/dataset/NSL-KDD/KDDTest+.txt', header=None)
df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']
qp.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']
df = df.drop('difficulty_level', 1)  # we don't need it in this project
qp = qp.drop('difficulty_level', 1)
df.isnull().values.any()
qp.isnull().values.any()
cols = ['protocol_type', 'service', 'flag']


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


# Merging train and test data
combined_data = pd.concat([df, qp])
# Applying one hot encoding to combined data
combined_data = one_hot(combined_data, cols)


# Function to min-max normalize
def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy()  # do not touch the original df
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# Dropping subclass column for training set
tmp = combined_data.pop('subclass')
new_train_df = normalize(combined_data, combined_data.columns)
# Fixing labels for training set
classlist = []
check1 = ("apache2","back","land","neptune","mailbomb","pod","processtable","smurf","teardrop","udpstorm","worm")
check2 = ("ipsweep","mscan","nmap","portsweep","saint","satan")
check3 = ("buffer_overflow","loadmodule","perl","ps","rootkit","sqlattack","xterm")
check4 = ("ftp_write","guess_passwd","httptunnel","imap","multihop","named","phf","sendmail","Snmpgetattack","spy","snmpguess","warezclient","warezmaster","xlock","xsnoop")

DoSCount = 0
ProbeCount = 0
U2RCount = 0
R2LCount = 0
NormalCount = 0

for item in tmp:
    if item in check1:
        classlist.append("DoS")
        DoSCount = DoSCount+1
    elif item in check2:
        classlist.append("Probe")
        ProbeCount = ProbeCount+1
    elif item in check3:
        classlist.append("U2R")
        U2RCount = U2RCount+1
    elif item in check4:
        classlist.append("R2L")
        R2LCount = R2LCount+1
    else:
        classlist.append("Normal")
        NormalCount = NormalCount + 1

new_train_df["Class"] = classlist

y_train = new_train_df["Class"]
# print(y_train.shape)
combined_data_X = new_train_df.drop('Class', 1)
# print(combined_data_X.shape)
oos_pred = []
dr = []
F1 = []
fpr = []
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
kfold.get_n_splits(combined_data_X, y_train)


class EQLv2(tf.keras.losses.Loss):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=5,  # 1203 for lvis v1.0, 1230 for lvis v0.5
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

    input = Input(shape=(122,1), dtype='float32')
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
    output = Dense(5)(flatten)
    model = Model(inputs=input, outputs=output)
    eqloss = EQLv2()
    model.compile(optimizer='adam', loss=eqloss, metrics=['accuracy'])
    model.summary()
    return model


model = build_model()
for train_index, test_index in kfold.split(combined_data_X,y_train):
    train_X, test_X = combined_data_X.iloc[train_index], combined_data_X.iloc[test_index]
    train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]
    # print(train_X.shape, test_X.shape)
    x_columns_train = new_train_df.columns.drop('Class')
    x_train_array = train_X[x_columns_train].values
    x_train_1=np.reshape(x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1))
    # print(x_train_1.shape)
    dummies = pd.get_dummies(train_y) # Classification
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


    model.fit(x_train_1, y_train_1,validation_data=(x_test_2,y_test_2), epochs=100, batch_size=256)

    pred = model.predict(x_test_2)
    pred = np.argmax(pred, axis=1)
    y_eval = np.argmax(y_test_2,axis=1)
    score = metrics.accuracy_score(y_eval, pred)
    f1 = metrics.f1_score(y_eval, pred, average='weighted')
    # recall = metrics.recall_score(y_eval, pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
    target_names = ['Dos', 'Normal', 'Probe', 'R2L', 'U2R']
    print(classification_report(y_eval, pred, target_names=target_names))
    cm = metrics.confusion_matrix(y_eval, pred)
    print(cm)
    tp = cm[0][0]+cm[2][2]+cm[3][3]+cm[4][4]
    fn = cm[0][1]+cm[2][1]+cm[3][1]+cm[4][1]
    fp = cm[1][0]+cm[1][2]+cm[1][3]+cm[1][4]
    tn = cm[1][1]
    DR = tp/(tp+fn)
    FPR = fp/(fp+tn)
    f1 = metrics.f1_score(y_eval, pred, average='weighted')
    oos_pred.append(score)
    F1.append(f1)
    dr.append(DR)
    fpr.append(FPR)
    print("ACC: {}".format(score))
    print("DR=recall:", DR)
    print("FPR:", FPR)
    print("f1:", f1)

print(oos_pred)
print(dr)
print(fpr)
print(F1)
