import torch
import joblib
import torch.nn as nn
import cnn_models
import pyttsx3
import numpy as np
import cv2
import argparse
import torch.nn.functional as Function
import time
import os
 
from torchvision import models
from gtts import gTTS


print(torch.__version__)
print(torch.cuda.is_available())

print(f"CUDA version: {torch.version.cuda}")
  
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")        
print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

data_lb = joblib.load('./predictions/data_lb.pkl')

translator_model = cnn_models.TranslatorCNN().cuda()
translator_model.load_state_dict(torch.load('./predictions/model.pth'))
print(translator_model)
print('Model loaded')

def sign_area(img):
    sign = img[100:324, 100:324]
    sign = cv2.resize(sign, (224,224))
    return sign

capture = cv2.VideoCapture(0)

if (capture.isOpened() == False):
    print('Error occurred while displaying the camera screen....')

#frame_width & frame_height
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

# creating a video output
out = cv2.VideoWriter('./predictions/predicted_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))


while(capture.isOpened()):
    # capturing the every frame
    ret, frame = capture.read()
    cv2.rectangle(frame, (100, 100), (324, 324), (20,34,255), 2)
    sign = sign_area(frame)

    image = sign
    
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).cuda()
    image = image.unsqueeze(0)
    
    outputs = translator_model(image)
    _, preds = torch.max(outputs.data, 1)
    
    cv2.putText(frame, data_lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)
    cv2.imshow('image', frame)
    out.write(frame)

 ########################################################################
    # # Initialize the engine
    # engine = pyttsx3.init()

    # # Set properties
    # engine.setProperty('volume', 0.9) # Volume 0-1

    # # Convert text to speech
    # text = lb.classes_[preds]
    # engine.say(text)
    # engine.save_to_file(text, 'output.mp3')
    # # engine.runAndWait()
 ###########################################################################
    # # time.sleep(0.09)

    # # Define the text you want to convert to speech
    # text = lb.classes_[preds]

    # # Create a gTTS object and specify the language
    # tts = gTTS(text=text, lang='en')

    # # tts.stream(text)

    # tts.save("out.mp3")

    # # Play the speech using your system's default media player
    # # os.system("start out.mp3")
 #############################################################################

    # press `q` to exit
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break


capture.release()

# closing video frame
cv2.destroyAllWindows()

