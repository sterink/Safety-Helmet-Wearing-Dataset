# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 22:23:35 2019

@author: czz
"""
import json
import matplotlib.path as mplPath

from gluoncv import model_zoo, data, utils

#from matplotlib import pyplot as plt
import mxnet as mx
import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--image', type=str, default='1.jpg',
                        help="image to test.")
    parser.add_argument('--network', type=str, default='yolo3_darknet53_voc',
                        #use yolo3_darknet53_voc, yolo3_mobilenet1.0_voc, yolo3_mobilenet0.25_voc 
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--short', type=int, default=416,
                        help='Input data shape for evaluation, use 320, 416, 512, 608, '                  
                        'larger size for dense object and big size input')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='confidence threshold for object detection')
    parser.add_argument('--url', type=str, default='rtsp://admin:admin@10.82.21.151/ch1/stream1',
                        help='video stream')
    parser.add_argument('--clip', type=str, default='clip_roi.json',
                        help='video stream')

    parser.add_argument('--gpu', action='store_false',
                        help='use gpu or cpu.')
    parser.add_argument('--dump', type=bool, default=False, help='save result into video or not')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()


    out = open(args.clip,'r')
    clip = json.load(out)['roi']
    bbPath = mplPath.Path(clip)


    if args.gpu:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
        
    net = model_zoo.get_model(args.network, pretrained=False)
    
    classes = ['hat', 'person']
    for param in net.collect_params().values():
        if param._data is not None:
            continue
        param.initialize()
    net.reset_class(classes)
    net.collect_params().reset_ctx(ctx)
    
    if args.network == 'yolo3_darknet53_voc':
        net.load_parameters('models/darknet.params',ctx=ctx)
        print('use darknet to extract feature')
    elif args.network == 'yolo3_mobilenet1.0_voc':
        net.load_parameters('models/mobilenet1.0.params',ctx=ctx)
        print('use mobile1.0 to extract feature')
    else:
        net.load_parameters('models/mobilenet0.25.params',ctx=ctx)
        print('use mobile0.25 to extract feature')

    cap = cv2.VideoCapture(args.url)

    if args.dump:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter('helmet.avi',fourcc, 20.0, sz,True)

    while True:
        ret, frame = cap.read()
        if ret==False:
            break
            

        # cv2.imshow('image', frame)
        # cv2.waitKey(1)
        # continue
        # # frame = cv2.imread(args.image)

        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        imgs = [mx.nd.array(img,dtype=np.uint8)]
        x, orig_img = data.transforms.presets.yolo.transform_test(imgs, short=args.short)
        scale = img.shape[0]/orig_img.shape[0]
    
        x = x.as_in_context(ctx)
        box_ids, scores, bboxes = net(x)

        # set clip box
        points = bboxes[0].asnumpy()
        points = points.reshape(-1)
        points = points[points!=-1].reshape(-1,2)

        active = bbPath.contains_points(points)
        print('active corners ',points[active])

        ax = utils.viz.cv_plot_bbox(frame, bboxes[0], scores[0], box_ids[0], scale=scale, class_names=net.classes,thresh=args.threshold)
        if args.dump:
            out.write(frame)

        # cv2.imshow('image', cv2.resize(frame,None,fx=0.25,fy=0.25))
        # cv2.waitKey(1)
        # cv2.imwrite(frame.split('.')[0] + '_result.jpg', orig_img[...,::-1])
    

cap.release()
if args.dump:
    out.release()

cv2.destroyAllWindows()
