# -*- coding: utf-8 -*-
import argparse

import numpy as np
import cv2 as cv
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

roi = []

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--url', type=str, default='rtsp://admin:admin@10.82.21.151/ch1/stream1',
                        help='video stream')
    parser.add_argument('--clip', type=str, default='clip_roi.json',
                        help='video stream')

    args = parser.parse_args()
    return args

def draw_canvas():
    global canvas
    canvas = frame.copy()
    for e in roi:
        ix,iy = e[0],e[1]
        cv.circle(canvas,(ix,iy),5,(0,0,255),-1)

    for id in range(len(roi)-1):
        cv.line(canvas,(roi[id][0],roi[id][1]),(roi[id+1][0],roi[id+1][1]),(0,255,255))

# mouse callback function
def draw_roi(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        roi.append([x,y])
    draw_canvas()
    # elif event == cv.EVENT_MOUSEMOVE:
    #     if drawing == True:
    #         if mode == True:
    #             cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
    #         else:
    #             cv.circle(img,(x,y),5,(0,0,255),-1)
    # elif event == cv.EVENT_LBUTTONUP:
    #     drawing = False
    #     if mode == True:
    #         cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
    #     else:
    #         cv.circle(img,(x,y),5,(0,0,255),-1)


if __name__ == '__main__':
    args = parse_args()

    cap = cv.VideoCapture(args.url)
    ret, frame = cap.read()

    frame = cv.imread('image/1.jpg')
    canvas = frame.copy()

    
    # # comment the below line if using stream
    # frame = np.zeros((512,512,3), np.uint8)

    cv.namedWindow('image')
    cv.setMouseCallback('image',draw_roi)
    while(1):
        cv.imshow('image',canvas)
        k = cv.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
    cv.destroyAllWindows()
    
    import json
    import matplotlib.path as mplPath
    
    out = open(args.clip,'w')
    json.dump({'roi':roi[:-1]}, out, indent = 6) 
    out.close()