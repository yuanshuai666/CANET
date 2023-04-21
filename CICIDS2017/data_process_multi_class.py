import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.model_selection import train_test_split
# from imblearn.under_sampling import RandomUnderSampler

data1 = pd.read_csv('/Data1/shuaiYuan/dataset/CICIDS2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
data2 = pd.read_csv('/Data1/shuaiYuan/dataset/CICIDS2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
data3 = pd.read_csv('/Data1/shuaiYuan/dataset/CICIDS2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv')
data4 = pd.read_csv('/Data1/shuaiYuan/dataset/CICIDS2017/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv')
data5 = pd.read_csv('/Data1/shuaiYuan/dataset/CICIDS2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
data6 = pd.read_csv('/Data1/shuaiYuan/dataset/CICIDS2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
data7 = pd.read_csv('/Data1/shuaiYuan/dataset/CICIDS2017/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv')
data8 = pd.read_csv('/Data1/shuaiYuan/dataset/CICIDS2017/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv')
data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8], axis=0, ignore_index=True, sort=False)



# Zero the missing value (nan)

data = data.fillna(value=0)
for i in range(len(data['Flow Bytes/s'])):
    if data['Flow Bytes/s'][i] == 'Infinity':
        # print(i)
        data['Flow Bytes/s'][i] = '1040000001'

for i in range(len(data[' Flow Packets/s'])):
    if data[' Flow Packets/s'][i] == 'Infinity':
        # print(i)
        data[' Flow Packets/s'][i] = '2000001'

data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
data_ = data.drop([' Label'], axis=1)
data_ed = data_.values
# StandardScaler/
data_ed = StandardScaler().fit_transform(data_ed)

L = data[' Label']
L = L.values
i = 0
for i in range(L.shape[0]):
    if L[i] == "BENIGN":
        L[i]=0
    elif L[i] == "DoS Hulk":
        L[i]=1
    elif L[i] == "PortScan":
        L[i]=2
    elif L[i] == "DDoS":
        L[i]=3
    elif L[i] == "DoS GoldenEye":
        L[i]=4
    elif L[i] == "FTP-Patator":
        L[i]=5
    elif L[i] == "SSH-Patator":
        L[i]=6
    elif L[i] == "DoS slowloris":
        L[i]=7
    elif L[i] == "DoS Slowhttptest":
        L[i]=8
    elif L[i] == "Bot":
        L[i]=9
    elif L[i] == "Web Attack Brute Force":
        L[i]=10
    elif L[i] == "Web Attack XSS":
        L[i]=11
    elif L[i] == "Infiltration":
        L[i]=12
    elif L[i] == "Web Attack Sql Injection":
        L[i]=13
    elif L[i] == "Heartbleed":
        L[i]=14

        i=i+1

label = L.reshape(L.shape[0], 1)



# print(data_ed.shape)          # (2827876, 78)
# print(Counter(L).items())
# (0, 2271320), (3, 128025), (2, 158804), (9, 1956), (12, 36), (10, 1507),
# (11, 652), (13, 21), (5, 7935), (6, 5897), (7, 5796), (8, 5499), (1, 230124), (4, 10293), (14, 11)
# print(label.shape)   # (2827876, 1)

np.save("./data/label.npy", label)
