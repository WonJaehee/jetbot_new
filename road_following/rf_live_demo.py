gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)480, height=(int)360, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)480, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
#gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

video_path = '../record/line.mp4'
import cv2
import numpy as np
from Adafruit_MotorHAT import Adafruit_MotorHAT

import torchvision
import torch

import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
	
angle = 0.0
angle_last = 0.0
steering_value = 0.0
steering_gain_value = 0.0
steering_bias_value = 0.0

def executeModel(image):
    global angle, angle_last

    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0

    print(x)
    print(y)
    print('---')
    '''
    angle = np.arctan2(x, y)
    pid = angle * steering_value + (angle - angle_last) * steering_dgain_value
    angle_last = angle
    
    steering_value = pid + steering_bias_value
    
    left_motor_value = max(min(speed_value + steering_value, 1.0), 0.0)
    right_motor_value = max(min(speed_value - steering_value, 1.0), 0.0)	
    '''

def imageCopy(src):
    return np.copy(src)
	
def imageProcessing(output):
    executeModel(output)
    return output

def Video(openpath, savepath = None):
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)
    print('--')

    out = None

    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    #cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    linecolor1 = (0,240,240)
    linecolor2 = (230,0,0)

    try:
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Our operations on the frame come here

                frame = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                output = imageProcessing(frame)
                #frame = np.copy(frame)			
                #im = cv2.line(im, (112, 0), (112, 224), linecolor1, 5, cv2.LINE_AA)			

                cv2.imshow("Input", frame)			

            else:
                break
            # waitKey(int(1000.0/fps)) for matching fps of video
            if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:  
        print("key int")
        all_stop()
        cap.release()
        cv2.destroyAllWindows()
        return

    # When everything done, release the capture
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    return
   
#if __name__=="__main__":
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
#model.load_state_dict(torch.load('best_steering_model_xy.pth'))
model.load_state_dict(torch.load('best_model_xy.pth'))
device = torch.device('cuda')
model = model.to(device)
model = model.eval().half()

#robot = Robot()
Video(gst_str)
#robot.stop()
