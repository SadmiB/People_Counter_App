"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
LABELS = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic', 11: 'fire', 13: 'stop', 14: 'parking', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports', 38: 'kite', 39: 'baseball', 40: 'baseball', 41: 'skateboard', 42: 'surfboard', 43: 'tennis', 44: 'bottle', 46: 'wine', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted', 65: 'bed', 67: 'dining', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy', 89: 'hair', 90: 'toothbrush', 0: 'None'}


# video frames intrevals with possitive detections
VIDEO_FRAMES = [(62,197),(229,447),(504,696),(747,867),(925,1196),(1237,1360)]

# number of the frames with possitive detections
true_detection =  sum([interval[1] - interval[0] + 1 for interval in VIDEO_FRAMES])



def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()

    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL) 
    
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    single_image_mode = False
    cur_req_id =  0
    last_count = 0
    total_count = 0
    start_time = 0
    avg_duration = 0
    model_detections = 0
    j = 0
    
    waiting = 0
    detecting = 0

    # Initialise the class
    infer_network = Network()
   
    ### Load the model through `infer_network` ###
    
    n, c, h, w = infer_network.load_model(args.model, args.device, cur_req_id, 1, 1, args.cpu_extension)[1]
    
    ### Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
    else:
        input_stream = args.input
    
    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)
    
    if not cap.isOpened():
        log.error('Please check your video source!')
    
    global init_w, init_h, prob_thresh
    
    # Set Probability threshold for detections
    
    prob_thresh = args.prob_threshold
    init_w = int(cap.get(3))
    init_h = int(cap.get(4))
    i = 0
    ### Loop until stream is over ###
    while cap.isOpened():
        ### Read from the video capture ###
        flag, frame = cap.read()
        i += 1
        if not flag:
            break;
        
        key_pressed = cv2.waitKey(60)
        
        
        ### Pre-process the image as needed ###
        image = preprocessing(frame, n, c, h, w)
        
        inf_time = time.time()
        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(cur_req_id, image)
        
        ### Wait for the result ###
        if infer_network.wait(cur_req_id) == 0:
            
            inf_duration = time.time() - inf_time
          
            
            if i == 1:
                avg_duration = inf_duration
            else:
                avg_duration = (inf_duration + avg_duration)/2
            ### Get the results of the inference request ###
           
            result = infer_network.get_output(cur_req_id)
                
            ### Extract any desired stats from the results ###
            frame, cur_count = get_ssd_out(frame, result)
            
            if cur_count == 0:
                detecting = 0
                waiting += 1
                            
            else:
                waiting = 0
                detecting += 1
                for interval in VIDEO_FRAMES:
                    if interval[0] <= i <= interval[1]:
                        model_detections+=1
                        break
                            
            
            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            inf_time_msg = "Infer time: {:.3f}ms"\
                               .format(inf_duration * 1000)
            cv2.putText(frame, inf_time_msg, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            
            avg_time_msg = "Avg infer time: {:.3f}ms"\
                               .format(avg_duration * 1000)
            cv2.putText(frame, avg_time_msg, (15, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            
            
            count_msg = "Current count: {}".format(cur_count)
            cv2.putText(frame, count_msg, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            
            if cur_count > last_count and detecting > 15:
                start_time = time.time() 
                total_count += (cur_count - last_count)
                client.publish('person', json.dumps({'total': total_count}))
                j = i
                
            
            if cur_count < last_count and waiting > 15:
                duration = int(time.time() - start_time - (i-j)/34) # 34 is the number of frames per second
                client.publish('person/duration', json.dumps({'duration': duration}))
               
                
                
            client.publish('person', json.dumps({'count': cur_count}))
            
            if single_image_mode:
                total_count = cur_count
            total_count_msg = "Total count: {}".format(total_count)
            
            cv2.putText(frame, total_count_msg, (15, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            
            
            if waiting > 15 or detecting > 15: 
                last_count = cur_count
                waiting = 0
                detecting = 0
            
            if key_pressed == 27:
                break
            
        ### Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###     
        if single_image_mode:
            cv2.imwrite('images/output_image.jpg', frame)
            
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    #print(avg_time_msg)
    #print("Accuracy: {:.3f}".format(model_detections/true_detection))
            
def preprocessing(input_image, n, c, h, w):
    '''
    Given an input image, height and width:
    - Resize to height and width
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = cv2.resize(input_image, (w, h))
    image = image.transpose((2,0,1))
    image = image.reshape((n, c, h, w))

    return image


def get_ssd_out(image, result):
    """
        Given an image, result of inference:
        return the current count of people
        and the frame with bound boxes
    """
    cur_count = 0
    
    for obj in result[0][0]:
        
        if int(obj[1]) != 1:
            continue
        
        if obj[2] > prob_thresh:
            xmin = int(obj[3] * init_w)
            ymin = int(obj[4] * init_h)
            xmax = int(obj[5] * init_w)
            ymax = int(obj[6] * init_h)
            
            cv2.rectangle(image, (xmin, ymin), (xmax,ymax), (0, 255, 0), 1)
            cv2.putText(image, LABELS[int(obj[1])], (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cur_count += 1
            
    return image, cur_count

    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
