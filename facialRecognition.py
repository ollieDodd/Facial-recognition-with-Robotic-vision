import imghdr
from tensorflow import keras
import cv2
import numpy as np
import os
import socket

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import time
import matplotlib.pyplot as plt
import seaborn as sns

#This function gets the names of all users in the dataset
def users():
    users = []
    dir1 = '/pImages'
    for files in os.listdir(dir1):
        file = str(files)
        users.append(file)
    return users
#This function detects faces 
def detection(img):
    dFace = haars.detectMultiScale(img,1.3,5)
    if (len(dFace)==1):
        x,y,w,h = dFace[0]
        img = img[y:y+h, x:x+w]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (112,92))
        return img
    else:
        return None
#This function allows for recognition
def recognition(model,target):
    im_rows=112
    im_cols=92
    im_shape=(im_rows, im_cols, 1)
    d.append(target)
    target = np.asarray(d)
    target = np.array(target,dtype='float32')/255
    target = target.reshape(-1,112,92,1)
    yhat = model.predict(target)
    return yhat[0]
#This function gets the averages of the predictions
def averageclassification(r,i):
    averages = []
    for x in r:
        averages.append(x / i)
    print(averages)
    max = np.argmax(averages)
    print(averages[max])
    if averages[max] > 5:
        return max 
    else: 
        return -1
#This function loads the model into the dataset
def initRecognition():
    model = load_model('/Model.h5')
    r = []
    data = []
    user = users()
    for i in user:
        r.append(0)
    r = np.asarray(r)
    i = 0
    dir1 = 'Input'

    for filename in os.listdir(dir1):
        if filename.endswith(".png"):
            f = dir1+"/"+filename
            img= cv2.imread(f)
            target = detection(img)
            if target is not None:
                data.append(target)
    print("images collected")
    for img in data:
        f = recognition(model, img)
        f = np.asarray(f)
        r = np.add(r,f)
        i+=1
    p = averageclassification(r,i)

    pS = str(user[p-1])
    response = pS

    print(response)
    return response
#This function trains the model
def training():
    dir1 = '/pImages'
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_valid = []
    y_valid = []
    count = 0
    i = -1
    for files in os.listdir(dir1):
        file = str(files)
        iFile = dir1+"/"+file
        count = 0
        i+=1
        print(len(x_train))
        for filename in os.listdir(iFile):
            if filename.endswith(".png"):
                if count <= 66:
                    f = iFile+"/"+filename
                    img= cv2.imread(f)
                    target = detection(img)
                    if target is not None:
                        x_train.append(target)
                        y_train.append(i)
                        count+=1
                elif count > 66 and count <= 94 :
                    f = iFile+"/"+filename
                    img= cv2.imread(f)
                    target = detection(img)
                    if target is not None:
                        x_test.append(target)
                        y_test.append(i)
                        count+=1
                elif count > 94 and count < 100:
                    f = iFile+"/"+filename
                    img= cv2.imread(f)
                    target = detection(img)
                    if target is not None:
                        x_valid.append(target)
                        y_valid.append(i)
                        count+=1

    x_train = np.array(x_train,dtype='float32')/255
    x_train = x_train.reshape(-1,10304)

    x_test = np.array(x_test,dtype='float32')/255
    x_test = x_test.reshape(-1,10304)

    x_valid = np.array(x_valid,dtype='float32')/255
    x_valid = x_valid.reshape(-1,10304)


    print('x_train : {}'.format(x_train[:]))
    print('Y-train shape: {}'.format(y_train))
    print('x_test shape: {}'.format(x_test.shape))

 

    im_rows=112
    im_cols=92
    im_shape=(im_rows, im_cols, 1)

    #change the size of images
    x_train = x_train.reshape(x_train.shape[0], *im_shape)
    x_test = x_test.reshape(x_test.shape[0], *im_shape)
    x_valid = x_valid.reshape(x_valid.shape[0], *im_shape)

    n_classes = len(np.unique(y_train))



    cnn_model= Sequential([
    Conv2D(filters=128, kernel_size=(8,8), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=256, kernel_size=(8,8), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=256, kernel_size=(5,5), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=512, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.10),
    Flatten(),
    Dense(2024, activation='relu'),
    Dropout(0.25),
    Dense(1024, activation='relu'),
    Dropout(0.25),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(n_classes, activation='softmax'),
    ])
    ttime0 = time.time()
    cnn_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(lr=0.0001),
        metrics=['accuracy']
    )
    history=cnn_model.fit(
        np.array(x_train), np.array(y_train), batch_size=512,
        epochs=20, verbose=2,
        validation_data=(np.array(x_valid),np.array(y_valid)),
        #callbacks= myCallBacks,
    )
    (cnn_model.summary())
    ttime1 = time.time()

    scor = cnn_model.evaluate( np.array(x_test),  np.array(y_test), verbose=0)
    pred = np.argmax(cnn_model.predict(x_test),axis = 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    print('test los {:.4f}'.format(scor[0]))
    print('test acc {:.4f}'.format(scor[1]))
    confusion = tf.math.confusion_matrix(labels=y_test, predictions= pred,)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(confusion, annot=True, fmt='d')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    print('test los {:.4f}'.format(scor[0]))
    print('test acc {:.4f}'.format(scor[1]))
    print("Total  training time")
    print(ttime1-ttime0)
    modelFile = '/userModels'
    fileType = '.h5'
    modelFile = modelFile + '/'+"Model"+fileType
    cnn_model.save(modelFile)
#Loads the haars cascade 
haars = cv2.CascadeClassifier("haars_face.xml')
#loads the directory for the images for recogniton
dir = '/Input'
#The following code handles messages between the server and clinet
d = []
msg =""
serv = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
serv.bind(('localhost',8080))
serv.listen(5)
while True:
    conn, addr = serv.accept()
    from_client = ""
    while True:
        os.system("cls")
        msg = conn.recv(4096)
        msg = msg.decode()
        if msg == "Recognition":
            del d[:]
            response = initRecognition()
            conn.send(response.encode())  
            break
        elif msg == "Training":
            training()
            response = "Facial data added to Facial recognition set"
            conn.send(response.encode())
            break
    conn.close()
    
