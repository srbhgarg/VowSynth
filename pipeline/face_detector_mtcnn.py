
#This is the main preprocessing file. This file saves face coordinates in separate file. output file: [filename]_face_coord.npy
import cv2
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import scipy.io as scio
import os
import numpy as np
import matplotlib.pyplot as plt
#for dlib to use GPU and maybe get a better speed
import dlib.cuda as cuda;
import dlib
import sys
print(cuda.get_num_devices())
dlib.DLIB_USE_CUDA = True
print(dlib.DLIB_USE_CUDA)

#[stylelist, vowellist, subjlist, filelist] = np.load('./results/demographics.npy')

outdir='./results/face/'
#os.mkdir(outdir)
filename = sys.argv[1]
coord=open(filename,"r");
filelist=coord.readlines()
for j in range(np.shape(filelist)[0]):
    i=j
    filelist[i]=filelist[i].rstrip('\n')
    vfile=filelist[i] +'_both.wmv'
    basename = os.path.basename(filelist[i])
    print(vfile)
    vid = cv2.VideoCapture(vfile)
    detector = MTCNN()
    while(vid.isOpened()):
        retval, image = vid.read()
        frame_num=vid.get(cv2.CAP_PROP_POS_FRAMES)
        if(frame_num==1):
            faces = detector.detect_faces(image)
            if(len(faces)>0):
                x,y,width,height =faces[0]['box']
                
                nx,ny=faces[0]['keypoints']['nose']
                lmx,lmy=faces[0]['keypoints']['mouth_left']
                rmx,rmy=faces[0]['keypoints']['mouth_right']
                lex,ley=faces[0]['keypoints']['left_eye']
                rex,rey=faces[0]['keypoints']['right_eye']
                # in case of negative values when full face in not in the image
                x=max(0,x)
                y=max(0,y)
            else:
                print("================================ ")
                print("Error: Face not detected !!!!!!! ")
                print("================================ ")
        else:
            vid.release()
            print("Saving: ", vfile)
            face_file= outdir+basename+'_face_coord'                       
            np.save(face_file,[y,min(y+height+100,np.shape(image)[0]-1),x-100,min(x+width+100,np.shape(image)[1]-1),  nx,ny,lmx,lmy,rmx,rmy,lex,ley,rex,rey])
            imfile= outdir+ basename+'_face.jpg'                       
            cv2.imwrite(imfile, image[y:min(y+height+100,np.shape(image)[0]-1),x:min(x+width,np.shape(image)[1]-1),:])
            
            #plt.imshow(image[y:y+height+100,x:x+width,:])
            #plt.show()


