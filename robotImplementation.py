#! /usr/bin/env python
# -*- encoding: UTF-8 -*-
import qi
import argparse
import sys
import time
from PIL import Image 
import cv2
import numpy as np
import os
import socket
from naoqi import ALProxy
from math import radians
haars = cv2.CascadeClassifier("haars_face.xml") #Loads the Haars cascade 
ip = "10.5.36.18" # Loads the IP for pepper
#THis function accesses peppers camera and stores the image as an OpenCV image
def getImage(session):
 
    video_service = session.service("ALVideoDevice")
    resolution = 2    # VGA
    colorSpace = 11   # RGB
    videoClient = video_service.subscribe("python_client", resolution, colorSpace, 5)

    t0 = time.time()
    naoImage = video_service.getImageRemote(videoClient)
    t1 = time.time()
    print ("acquisition delay "), t1 - t0

    video_service.unsubscribe(videoClient)

    imageWidth = naoImage[0]
    imageHeight = naoImage[1]
    array = naoImage[6]
    image_string = str(bytearray(array))

    im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string)
    numpy_pil = np.array(im)  
    opencvimg = cv2.cvtColor(numpy_pil, cv2.COLOR_RGB2BGR)
    return opencvimg
#This function allows for face detection
def detection(img):
    if img is not None:
        dFace = haars.detectMultiScale(img,1.3,5)
        if (len(dFace)==1):
            x,y,w,h = dFace[0]
            img = img[y:y+h, x:x+w]
            img = cv2.resize(img, (200,200))
            return True
        else:
            return False   
#This function saves an image to a chosen path
def saveImg(img,i,type):
    path = ("C:/Users/olive/Documents/Project/Collecting_data/Input")
    sI = str(i)
    file= sI+u+type
    cv2.imwrite(os.path.join(path,file),img)
    i+= 1
    return i 
#This function adds data to allow for enrolement
def addingData(name):
    tts.say("The system will now take 100 photos, please make sure the lighting conditions are bright and your facing your camera with minimal movement")
    i = 1
    while i == 1:
        i2 = 0
        tts.say("Please press y when ready, if you wish to stop press n - ")
        d = raw_input("")
        if d == "y":
            i2 = 0
            type = ".png"
            path = ("C:/Users/olive/Documents/Project/Collecting_data/pImages/")
            path = os.path.join(path,name)
            os.mkdir(path)
            while i2 < 100:
                frame = getImage(session)
                print("Got image")
                if detection(frame) == True:
                    print("Face detected")
                    sI = str(i2)
                    file= name+sI+type
                    cv2.imwrite(os.path.join(path,file),frame)
                    i2+=1
                else: print("Face not detected")
            tts.say("100 images captured")
            break
        elif d == "n":
            break
        else:
            i = 1
#This function checks initial face detection
def faceDetected():
    tts.say("face detected")
    tts.say("Is your facial data within the system")
    response = raw_input()
    if response == "y":
        startRecognition()
    if response == "n":
        #tts.say("Have a good day")
        startTraining()

#This function allows for facial recognition
def startRecognition():
    client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client.connect(("localhost",8080))
    type = ".png"
    i = 0 
    i2 = 0
    while i2< 20:
        frame = getImage(session)
        if detection(frame) == True:
            print("Face detected")
            i = saveImg(frame,i,type)
            i2+=1
        else: 
            print("No face detected")
    data = "Recognition"
    client.send(data.encode())
    
    response = client.recv(4096)
    client.close()
    response = response.decode()
    response = str(response)
    if response != "User not recognised":
        tts.say("Hello")
        tts.say(response)
        tts.say("Have a good day")
#This function allows for enrolement
def startTraining():
        d= 1
        while d == 1:
            d = 0
            tts.say("User not recognised in the system, would you like to add facial structure to the system. Please input y or n")
            decison = raw_input("")
            if decison == "y":
                tts.say("What is your name?")
                name = raw_input("")
                addingData(name)
                data = "Training"
                client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                client.connect(("localhost",8080))
                client.send(data.encode())

                response = client.recv(4096)
                response = response.decode()
                client.close()
                response = str(response)
                #tts = ALProxy("ALTextToSpeech", "10.5.24.52", 9559)
                tts.say(response)
                
            elif decison== "n":
                print("End")
            else:
                z = 1
#This function rotates the robot 180 degrees 
def moveRotate(m,a):
    m.moveInit()
    m.post.moveTo(0,0,radians(a))
    m.waitUntilMoveIsFinished()
#This function moves pepper 1 metre and whilst looking for faces 
def moveStraight(m,session):
    m.moveInit()
    m.post.moveTo(1,0,radians(0))
    while m.moveIsActive() == True:
        img = getImage(session)
        if detection(img) == True:
            m.stopMove()
            #tts.say("Face found")
            faceDetected()
    m.waitUntilMoveIsFinished()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default=ip,
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    m = ALProxy("ALMotion", ip, 9559)
    tts = ALProxy("ALTextToSpeech", ip, 9559)
#Gets the robot to navigate
    count = 0
    while count < 5:
        moveStraight(m,session)
        moveRotate(m,180)
        moveStraight(m,session)
        moveRotate(m,-180)

        count+=1
    moveRotate(m,180)
