# Facial-recognition-with-Robotic-vision
## Summary
  The following repo holds a code implementation of a facial recognition system using Soft Robotics Bank Robotics pepper. The facial recognition model is powered by Tensorflow and Keras, where 100 images of the user need to be inputted and a model is trained. This implementation provides algorithms to allow Pepper to navigate its environment, start interaction with a person and either store facial images for model training or takes facial images for recognition. Through testing the more facial data added to the system the higher accuracy the program will return.
 The robot implementation of the artefact uses python 2.7 and the machine learning implementation uses python 3.9. Sockets will be used to bridge the two systems together to allow them to work together. Socket programming is a way of connecting two nodes on a network to communicate with each other. 
  
## How to use
  For initial use of the system no model is present in the GitHub repository so on first interaction with pepper a model needs to be trained by telling pepper your facial data is not in the system. When images are captured, this is stored with researcher gathered images to train the model.
  When pulling the GitHub make sure all file directories are correct.
  The robotic implementation needs to be ran in python 2.7. The Facial recognition implementation needs to be ran in python 3.9.

  
## Video Demonstration 
A video demonstration can be found here (https://www.youtube.com/watch?v=gxuvKegRUvI&ab_channel=OliverDodd)

## Libraries required 
 - Tensorflow
 - OpenCV
 - Numpy
 - OS 
 - Sockets
 - Time
 - Matplotlib
 - seaborn
 - PIL
 - QI
 - Argparse
 - Sys
 - Math
 - Naoqi

### This artefact was created for a Dissertation project for the Univeristy of Lincoln 
### Final grade - 78
