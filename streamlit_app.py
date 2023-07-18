import streamlit as st
import cv2
import tempfile
from SpatioTemporalAutoEncoder import loadModel
import numpy as np
import imutils
import time

def mean_squared_loss(x1,x2):
    diff=x1-x2
    a,b,c,d,e=diff.shape
    n_samples=a*b*c*d*e
    sq_difference=diff**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples
    return mean_distance

model = loadModel()
model.load_weights('weights')


f = st.file_uploader("""#Upload video here""")

if f is not None: 
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())


    cap = cv2.VideoCapture(tfile.name)

    all_frames = []
    stframe = st.empty()

    while cap.isOpened():
        imDump=[]
        ret,frame=cap.read()
        no_more_frames = False
        for i in range(10):
            ret,frame=cap.read()
            if(ret == True):
                image = imutils.resize(frame,width=700,height=600)
                frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
                gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
                gray=(gray-gray.mean())/gray.std()
                gray=np.clip(gray,0,1)
                imDump.append(gray)
            else: 
                no_more_frames = True
                break 
            
        imDump=np.array(imDump)
        imDump.resize(227,227,10)
        imDump=np.expand_dims(imDump,axis=0)
        imDump=np.expand_dims(imDump,axis=4)
        output=model.predict(imDump)
        loss=mean_squared_loss(imDump,output)
        print(loss) 
        if no_more_frames: 
            break
        if frame.any() == None:
            print("none")
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
        if loss>0.000419:
            print('Abnormal Event Detected')
            cv2.putText(image,"Abnormal Event",(100,80),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),3)
        stframe.image(image, channels='BGR')


    cap.release()
    cv2.destroyAllWindows()