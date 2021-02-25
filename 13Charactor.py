#B0843020 宥俞
#13位角色分類,請先將甘露寺蜜璃資料夾內的 "20 - 複製.jpg" 改為 "20.jpg",以免讀取檔案時有可能會報錯
#只需更改train_file_path、test_file_path、output_file_path的路徑即可
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras import preprocessing as kp
import os
import cv2

train_file_path = 'D:/VisualStudioCode/python-project/Ai/CNN homework/winter_vacation_exercise_process/' #train data的路徑,請自行更改
test_file_path  = 'D:/VisualStudioCode/python-project/Ai/CNN homework/winter_vacation_exercise_process/' #test data的路徑,請自行更改 
output_file_path = 'D:/VisualStudioCode/python-project/Ai/CNN homework/output.txt' #output的路徑,請自行更改
label_to_index = {
    'Ganlu Temple Mili': 1,'Hu Dieren': 2,'Immortal Kawamiya': 3,'Kamado Tanjiro': 4,'Kamen Nidouzi': 5,
    'Mouth flat Inosuke': 6,'Purgatory Kyojuro': 7,'Sorrowful Islet Xing Ming': 8,'Toki Touichiro': 9,
    'Tomioka Yoshiyuki': 10,'U Maru Tianyuan': 11,'Yihei Xiaoban': 12,'Zenyi': 13 }

#training :D
ans_label=list()
train_file_jpg=list()
allList = os.walk(train_file_path)
for root, dirs, files in allList:
    for dir in dirs:
        ans_label.append(str(dir))
    for file in files:
        train_file_jpg.append(str(root+"/"+file)) #取得每一個jpg檔案的路徑
x_Train=list()
y_Train=list()
y_TrainOneHot=list()
for data in train_file_jpg:
    img = cv2.imread(data) #讀取圖檔
    x_train = img.reshape(250,250,3).astype('float32')
    x_train_normalize = x_train / 255
    x_Train.append(x_train_normalize)
x_Train = np.array(x_Train)
for data in ans_label: #製作y_train的list
    y_train=[data]*22
    y_Train.extend(y_train)
y_TrainOneHot=pd.get_dummies(y_Train)
y_TrainOneHot = np.array(y_TrainOneHot)
"""
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(25,25),padding='same',input_shape=(250,250,3),activation='relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu')) 
model.add(MaxPooling2D(pool_size=(5,5)))
model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.5)) #Dropout
model.add(Dense(len(y_TrainOneHot[0]), activation='softmax')) 
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(x=x_Train,y=y_TrainOneHot,validation_split=0.1,epochs=20,batch_size=10,verbose=1)
model.save('13chractor-2.h5') #儲存模型
"""
#testing :D
model =keras.models.load_model("13chractor-0.069.h5") #讀取模型
test_file_jpg=list()
test_jpg=list()
allList = os.walk(test_file_path)
for root, dirs, files in allList:
    for file in files:
        test_jpg.append(str(file))
        test_file_jpg.append(str(root+"/"+file)) #取得每一個jpg檔案的路徑
x_Test=list()
ans=list()
for data in test_file_jpg:
    img = cv2.imread(data) #讀取圖檔
    x_test = img.reshape(250,250,3).astype('float32')
    x_test_normalize = x_test / 255
    x_Test.append(x_test_normalize)
x_Test = np.array(x_Test)
y_test = model.predict(x_Test) #預測test的結果
y_test = y_test.tolist()
for i in y_test:
    ans.append(ans_label[i.index(max(i))])
#output :D
with open(output_file_path, 'w', encoding='utf-8') as f:
    for i in range(len(ans)):
        f.write(test_jpg[i])
        f.write(', ')
        f.write(ans[i])
        f.write('\n')
    f.close()
