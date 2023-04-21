import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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
        L[i] = 0
    else:
        L[i] = 1

        i = i+1

label = L.reshape(L.shape[0], 1)


np.save("./data/data.npy", data_ed)
np.save("./data/binary_label.npy", label)
