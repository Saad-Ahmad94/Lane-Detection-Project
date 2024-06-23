
import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from tensorflow import keras
import importlib_resources




#Load the model in Keras 
model = keras.models.load_model(r'C:\Personal Data\Projects\Lane Detection Project\model.h5')

#Make the lanes class

class Lanes():
    def __init__(self,model):
        self.model = model
        self.recent_fit = [] #Contains the most recent predictions
        self.avg_fit = [] #Contains the avg predictions 
    
    # A function for preprocessing video frame by frame

    def road_line(self, image):
        small_img = imresize(image,(80,160,3)) 
        small_img = np.array(small_img) #Makes code efficient
        small_img = small_img[None,:,:,:] #Add extra dimension for storing prediction for every pixel

        prediction = model.predict(small_img)[0] * 255
        self.recent_fit.append(prediction)

        if len(self.recent_fit) > 5:
            self.recent_fit = self.recent_fit[1:]

        self.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

        blanks = np.zeros_like(self.avg_fit).astype(np.uint8)
        lane_drawn = np.dstack((blanks, self.avg_fit, blanks))

        lane_image = imresize(lane_drawn, (720, 1280, 3))
        result = cv2.addWeighted(image, 1, lane_image, 1, 0)
        
        return result

lanes = Lanes(model)
vid_input = VideoFileClip(filename=r'C:\Personal Data\Projects\Lane Detection Project\lanes_clip.mp4')
vid_output = 'lanes_clip_out.mp4'

vid_clip = vid_input.fl_image(lanes.road_line)
vid_clip.write_videofile(vid_output)



