#jetson tx2 onboard camera gstreamer string
#gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
gst_str = ("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, format=(string)YUY2,framerate=30/1 ! videoconvert ! video/x-raw,width=640,height=480,format=BGR ! appsink")

video_path = '../record/line.mp4'
import cv2
import numpy as np

import torchvision
import torch

import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image

from torch2trt import TRTModule 


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
	
angle = 0.0
angle_last = 0.0

def executeModel(image):
    global angle, angle_last
    #image = change['new']
    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0
    
    #x_slider.value = x
    #y_slider.value = y

    print(x)
    print(y)
    print('---')

    
    #speed_slider.value = speed_gain_slider.value
    
    #angle = np.arctan2(x, y)
    #pid = angle * steering_gain_slider.value + (angle - angle_last) * steering_dgain_slider.value
    #angle_last = angle
    
    #steering_slider.value = pid + steering_bias_slider.value
    
    #robot.left_motor.value = max(min(speed_slider.value + steering_slider.value, 1.0), 0.0)
    #robot.right_motor.value = max(min(speed_slider.value - steering_slider.value, 1.0), 0.0)	

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
    #if savepath is not None:
        #out = cv2.VideoWriter(savepath, fourcc, fps, (width, height), True)
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    #cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    linecolor1 = (0,240,240)
    linecolor2 = (230,0,0)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            frame = np.copy(frame)
            im = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)
            output = imageProcessing(im)
            #output = imageProcessing(frame)
            #im = cv2.line(im, (320, 0), (320, 480), linecolor1, 5, cv2.LINE_AA)
            #im = cv2.line(im, (0, 300), (640, 300), linecolor2, 5, cv2.LINE_AA)			
            im = cv2.line(im, (112, 0), (112, 224), linecolor1, 5, cv2.LINE_AA)
            im = cv2.line(im, (0, 140), (224, 140), linecolor2, 5, cv2.LINE_AA)			

            cv2.imshow("Input", im)			
            # Write frame-by-frame
            #if out is not None:
            #    out.write(output)
            # Display the resulting frame
            #cv2.imshow("Input", frame)
            #cv2.imshow("Output", output)
        else:
            break
        # waitKey(int(1000.0/fps)) for matching fps of video
        if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    return
   
#if __name__=="__main__":
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('tov_model_xy_trt.pth'))

device = torch.device('cuda')
model = model_trt.to(device)
model = model_trt.eval()

#robot = Robot()
Video(gst_str)
#robot.stop()
