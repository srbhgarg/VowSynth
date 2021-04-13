#run face_detector_mtcnn before running this file
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import io
import cv2
from SAND.san_api import SanLandmarkDetector
device = "cpu"
model_path = "SAND/checkpoint_49.pth.tar"
det = SanLandmarkDetector(model_path, device)


#these functions adjust the head size and also fixes any rotation in the head
def compute_alignment(image, rex, rey, lex, ley, dist2=175, debug=False):
    h, w = np.shape(image)[0], np.shape(image)[1]
    # Calculating a center point of the image
    center = (w // 2, h // 2)

    ##fix rotation
    delx = rex-lex
    dely = rey-ley
    angle=np.arctan(dely/delx)

    #converto degrees
    angle = (angle * 180) / np.pi

    M = cv2.getRotationMatrix2D(center, (angle), 1.0)


    ##fix scaling
    # calculate distance between the eyes in the image for scaling later
    dist1 = np.sqrt((delx * delx) + (dely * dely))

    #calculate the ratio
    ratio = dist1 / dist2
    dim = (int(w * ratio), int(h * ratio))
    return  M, dim
    if (debug):
        plt.imshow(resized)
        plt.show()

def apply_alignemnt(image, M, dim ):
    h, w = np.shape(image)[0], np.shape(image)[1]
    rotated = cv2.warpAffine(image, M, (w, h))
    resized = cv2.resize(rotated, dim)
    
    #resized_face = cv2.transform(points, M)
    return resized


align=True #align=True will align the frames so that eyes are horizontal and distance between the eyes is 175
           #align=False will skip the step 

debug=False
outdir='./results/face/'

os.mkdir('./results/san_align/')
filename = sys.argv[1]
coord=open(filename,"r");
Lines=coord.readlines()
for line in Lines:
    subj=line.rstrip('\n')
    basename = os.path.basename(subj)
    vfile=subj+'_both.wmv'
    vid = cv2.VideoCapture(vfile)
    lm=[]
    while(vid.isOpened()):
        retval, image = vid.read()
        if retval==False:
            vid.release()
            break
        frame_num=vid.get(cv2.CAP_PROP_POS_FRAMES)
        total_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        if(frame_num<=total_count):
            face_file= outdir+basename+'_face_coord.npy'                       
            if (not os.path.exists(face_file)):
                face_file=outdir+basename+'_both_face_coord.npy'
            rect=np.load(face_file)    
            top = rect[0]-70
            bottom = rect[1]-50
            left = rect[2]+80
            right = rect[3]-80
            
            #nx,ny,lmx,lmy,rmx,rmy,
            lex =rect[10]
            ley=rect[11]
            rex=rect[12]
            rey=rect[13]
            
            
            ## to align faces
            if (align==True):
                if(frame_num==1):
                    M, dim=compute_alignment(image, rex, rey, lex, ley)
                    print(dim)
                #points=np.float32([[left, top], [right, bottom]])
                aligned=apply_alignemnt(image, M, dim)
                if 0:
                    faces = detector.detect_faces(aligned)
                    x,y,w, h =faces[0]['box']
                    left=x 
                    top=y 
                    right=x+w 
                    bottom=y+h
            
            else:
                aligned=image
                
            
            landmarks, scores = det.detect(aligned, [left, top, right, bottom])
            lm.append(landmarks)
            #print(frame_num,total_count, np.shape(lm))
            for lmp in range(68):
                x1=landmarks[lmp,0]
                y1=landmarks[lmp,1]
                pos=(int(x1),int(y1))
                cv2.circle(image, pos, 5, color=(255, 0, 0))
            #plt.imshow(image[top:bottom,left:right,:])
        else:
            vid.release()
        #save first frame for debugging
        if(frame_num==int(total_count/2)):
            print(vfile, retval, frame_num)
            imfile='./results/san_align/'+basename+'_san.jpg'                       
            cv2.imwrite(imfile, image[top:bottom,left:right,:])
        
        if debug:
            image=aligned
            landmarks_indices=[*range(4,13),*range(48,68)] #5:13 is lower jaw and 49:68 is lips
            len=np.shape(landmarks_indices)[0]
            fig=plt.figure(figsize=(14, 14), dpi= 80, facecolor='w', edgecolor='k')
            for lmp in range(len):
                x1=landmarks[landmarks_indices[lmp],0]
                y1=landmarks[landmarks_indices[lmp],1]
                pos=(int(x1),int(y1))
                #color is in BGR format
                image=cv2.circle(image, pos, 5, color=(0, 255, 255), thickness=-1)
            #plt.subplot(3,3,s+1)
            plt.imshow(image)
            plt.show()
  
    out_file='./results/san_align/'+basename+'_san'
    #print(out_file," ", np.shape(lm))
    np.save(out_file,lm)





