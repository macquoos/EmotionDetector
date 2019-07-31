import numpy as np
import cv2
from os import path

from detectEmotion import detectEmotion


CLASSIFIER_NAME = "face_classifier.xml"

save2file = False


# Load the classifier model
from keras.models import load_model
model = load_model('emotion_classifier')



#%% 1. Create an object
video = cv2.VideoCapture(0)


faceCascade = cv2.CascadeClassifier(CLASSIFIER_NAME)

# Define the codec and create VideoWriter object
if save2file:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output1.avi',-1, 20.0, (int(video.get(3)),int(video.get(4))))

num_frames = 0;


while(video.isOpened()):
        
    num_frames += 1
    
    #3. Create a frame object
    (check, frame) = video.read()
    
    if check:
        npframe = np.array(frame)
        npframe = cv2.flip(npframe,1)
        gray = cv2.cvtColor(npframe, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30,30))
        
        # drawing shapes
        font = cv2.FONT_HERSHEY_PLAIN 
        for (x,y,w,h) in faces:
            cv2.rectangle(npframe, (x,y), (x+w, y+h), (0, 255, 0), 2)
            faceSubFrame = gray[y:y+h,x:x+w]
            
            # detect emotion with model classifier
            emotion, conf = detectEmotion(faceSubFrame,(128,128),model)
            
            cv2.putText(npframe,  emotion +':  ' + str(round(conf,2))  ,(x+15,y+h+30), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        
        
        
        if save2file:
            out.write(npframe)
            
        # 4. Show the npframe!
        cv2.imshow("fdsfds", faceSubFrame)
        cv2.imshow("Capturing", npframe)
        # 5. For press any key to out (ms)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break



print("Number of captured frames: {}".format(num_frames))

#2. Shutdown the camera
video.release()

if save2file:
    out.release()

cv2.destroyAllWindows()