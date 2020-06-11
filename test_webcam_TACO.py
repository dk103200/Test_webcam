import numpy as np
import tensorflow as tf
import cv2 as cv
from contexttimer import Timer
import glob
import os
import shutil
import csv
# cmt = lambda: int(round(time.time() * 1000))

# Pretrained classes in the model
# classNames = {0: 'background',
#               1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
#               7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
#               13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
#               18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
#               24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
#               32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
#               37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
#               41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
#               46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
#               51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
#               56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
#               61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
#               67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
#               75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
#               80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
#               86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

classNames = {0: 'background',
              1: 'Can', 2: 'Other', 3: 'Bottle', 4: 'Plastic bag + wrapper'}

# Read the graph.
with tf.gfile.FastGFile('ssd_mobilenet_v1_taco_2.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# cap = cv.VideoCapture('test_TACO.mp4')
#  list_image = os.listdir(path+folder)
path = './images/'
#os.mkdir('Detected')
with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    cont = True
    list_image = os.listdir(path)
    for image in list_image:
        # Read and preprocess an image.
        img = cv.imread(path+image)
        # ret, img = cap.read()
        rows = img.shape[0]
        cols = img.shape[1]    
    
        inp = cv.resize(img, (300, 300))
        # inp = inp/127.5
        # inp = inp - 1
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        # Run the model
        with Timer() as t:
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                            feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0],
                                                                        inp.shape[1], 3)})
        print("Session.run() time: {}".format(t.elapsed))

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
       
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.5:
                print('box', bbox[1],bbox[0])
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
                cv.putText(img, classNames[classId],
                            (int(x), int(y+.05*(y-bottom))),
                            cv.FONT_HERSHEY_SIMPLEX,
                            # (.005 * image_width),
                            1,
                            (50, 200, 50), 2)
                cv.putText(img, str(score),
                            (int(x-20), int(y+.05*(y-bottom)-50)),
                            cv.FONT_HERSHEY_SIMPLEX,
                            # (.005 * image_width),
                            1,
                            (50, 200, 50), 2)
                with open('./writeData.csv', mode='a') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    #way to write to csv file
                    writer.writerow([x,y,right,bottom,classNames[classId],score])
                path2 = './Detected/'+image+'_detected.jpg'
                cv.imwrite(path2,img)
                print (i)
                print  ('Save done!\n')
                cv.imshow('TensorFlow MobileNet-SSD', img)
                cv.waitKey() 
        

cv.destroyAllWindows()
