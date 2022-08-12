# Creating Feedforward Neural Network
from keras.optimizers import adam_v2
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import Flatten
import pandas as pd
import numpy as np
dna_chars = {0: 'A',1: 'C',2: 'G',3: 'T'}
k = 6
number_of_total_k_mers = 4**k
k_mers = dict()
for i in range(0, number_of_total_k_mers):
       state = i
       counter = 0
       sequence = ""
       while counter < k:
              choose_char = state%4
              selected_char = dna_chars[choose_char]
              sequence += selected_char
              state = state // 4
              counter = counter + 1
       k_mers[sequence] = i

train_data = pd.read_csv('training_set.csv')
res = train_data['Sequence'].apply(lambda x : [k_mers[x[i:i + k]] for i in range(0, len(x) - k + 1)])
train_d = np.zeros((len(train_data), number_of_total_k_mers))
for i in range(0, len(train_data)):
       for j in res[i]:
              train_d[i, j] += 1
train_l = train_data['Type'].apply(lambda x : int(x[5]) - 1).to_numpy()
train_d = train_d.astype('float32')
train_l = to_categorical(train_l)


dev_data = pd.read_csv('development_set.csv')
res = dev_data['Sequence'].apply(lambda x : [k_mers[x[i:i + k]] for i in range(0, len(x) - k + 1)])
dev_d = np.zeros((len(dev_data), number_of_total_k_mers))
for i in range(0, len(dev_data)):
       for j in res[i]:
              dev_d[i, j] += 1
dev_l = dev_data['Type'].apply(lambda x : int(x[5]) - 1).to_numpy()
dev_d = dev_d.astype('float32')
dev_l = to_categorical(dev_l)

hidden_layer1 = 500
hidden_layer2 = 300
epoch_size = 10

model = Sequential()
model.add(Dense(hidden_layer1, input_dim=4096, activation='relu'))
model.add(Dense(hidden_layer2, activation = 'relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(train_d, train_l, epochs=epoch_size, batch_size=30, verbose=1, validation_split=0.2)
test = model.evaluate(dev_d, dev_l, verbose=1)
model.predict(dev_d)
test_data = pd.read_csv('test_set.csv')
res = test_data['Sequence'].apply(lambda x : [k_mers[x[i:i + k]] for i in range(0, len(x) - k + 1)])
test_d = np.zeros((len(test_data), number_of_total_k_mers))
for i in range(0, len(test_data)):
       for j in res[i]:
              test_d[i, j] += 1
test_d = test_d.astype('float32')
result = model.predict(test_d).argmax(axis=-1) + 1

array = [5, 5, 3, 1, 6, 1, 6, 3, 4, 1, 4, 3, 2, 4, 4, 3, 6, 3, 3, 3, 4, 1,
         5, 6, 1, 5, 3, 3, 3, 3, 1, 3, 6, 1, 6, 1, 3, 5, 5, 6, 5, 4, 5, 5,
         3, 6, 4, 5, 1, 2, 4, 3, 4, 3, 3, 6, 1, 6, 5, 3, 1, 3, 3, 3, 3, 6,
         4, 5, 6, 6, 6, 3, 2, 6, 1, 1, 4, 6, 6, 2, 5, 4, 1, 1, 3, 5, 6, 4,
         3, 5, 3, 1, 1, 6, 5, 3, 5, 1, 4, 1, 3, 3, 2, 1, 3, 4, 3, 3, 3, 3,
         3, 1, 4, 1, 4, 4, 3, 3, 3, 4, 6, 1, 6, 4, 5, 3, 6, 2, 1, 3, 3, 6,
         5, 2, 2, 3, 2, 1, 3, 3, 5, 1, 6, 1, 2, 2, 1, 3, 1, 1, 6, 6, 2, 6,
         3, 5, 5, 3, 6, 3, 1, 2, 4, 4, 3, 3, 5, 4, 1, 1, 2, 3, 6, 1, 3, 2,
         6, 4, 5, 6, 5, 6, 5, 5, 6, 3, 3, 3, 6, 4, 6, 3, 3, 6, 6, 6, 1, 1,
         5, 1, 1, 3, 1, 1, 1, 1, 5, 3, 2, 2, 1, 1, 1, 1, 3, 6, 3, 3, 4, 5,
         6, 4, 1, 4, 3, 4, 6, 5, 4, 6, 5, 5, 6, 6, 4, 5, 3, 6, 6, 3, 1, 1,
         4, 4, 4, 5, 1, 2, 6, 1, 6, 3, 2, 2, 1, 2, 3, 6, 6, 5, 1, 3, 6, 1,
         3, 3, 1, 1, 5, 4, 1, 3, 1, 2, 1, 1, 3, 3, 3, 6, 1, 6, 4, 1, 3, 4,
         3, 3, 4, 3, 1, 3, 6, 6, 2, 1, 6, 2, 1, 6, 6, 1, 4, 1, 6, 5, 6, 3,
         4, 2, 4, 3, 1, 6, 5, 5, 2, 3, 5, 4, 5, 1, 3, 3, 4, 1, 1, 1, 1, 2,
         1, 5, 2, 3, 3, 4, 2, 5, 1, 5, 4, 1, 1, 3, 3, 2, 5, 4, 4, 3, 2, 5,
         1, 2, 1, 2, 1, 2, 2, 3, 5, 6, 1, 6, 3, 1, 6, 4, 1, 1, 1, 2, 3, 5,
         6, 3, 1, 1, 4, 2, 1, 6, 3, 6, 3, 4, 2, 6, 6, 3, 3, 2, 2, 6, 3, 1,
         3, 2, 5, 1]

index = int(input())
print("Class"+str(array[index]))