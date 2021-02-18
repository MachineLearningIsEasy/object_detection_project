import os
import datetime
import json
import requests
import configparser

import cv2

import numpy as np
import tensorflow as tf

def bb_intersection_over_union(boxA, boxB):
    '''
    IOU for person detection in alarm box.
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

if __name__=='__main__':
    # parameters
    
    # config
    config = configparser.ConfigParser()
    try:
        config.read('ini.conf')
    except Exception:
        print('[ERROR] ini.conf dont found.')

    path_to_video =  config['video']['path_to_video']
    video_filename = config['video']['video_filename']

    path_to_saved_model = config['tf_serving']['path_to_saved_model']
    tf_serving_url = config['tf_serving']['tf_serving_url']

    data_base_path = config['db']['data_base_path']

    seconds = int(config['main_work']['seconds'])
    iou_thresh = float(config['main_work']['iou_thresh'])
    score_tresh = float(config['main_work']['score_tresh'])

    USE_TF_SERVING = True if config['flags']['USE_TF_SERVING']=='True' else False
    WRITE_RESULT_VIDEO = True if config['flags']['WRITE_RESULT_VIDEO']=='True' else False

    # Init values
    stop_line_box = (0,0,100,100)
    video_writer = None
    video_out_file_path = None
    fourcc = None
    w_image, h_image = None, None
    detection_model = None

    # Init steps
    if WRITE_RESULT_VIDEO:
        video_out_file_path = os.path.join(path_to_video, '{}_out.avi'.format(video_filename))
    
    if not USE_TF_SERVING:
        detection_model = tf.saved_model.load(path_to_saved_model)

    print("[INFO] starting video stream...")
    # Open VideoCapture
    path_to_videofile = os.path.join(path_to_video, video_filename)
    vs = cv2.VideoCapture(path_to_videofile)
    # Open VideoWriter
    if WRITE_RESULT_VIDEO:
        # video_writer = cv2.VideoWriter()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Prepare for frames loop info
    fps = vs.get(cv2.CAP_PROP_FPS)
    multiplier = int(fps * seconds)
    frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    frameId = 0

    alarm_time_on_video = []
    # loop over the frames from the video stream
    while True:
        # read frame
        frame = vs.read()
        frameId = int(round(vs.get(1)))    
        # check to see if we have reached the end of the stream
        if frameId >= frame_count:
            break

        frame = frame[1]
        if frameId == 1: 
            if WRITE_RESULT_VIDEO:
                video_writer = cv2.VideoWriter(video_out_file_path, fourcc, 1.0, (frame.shape[1],frame.shape[0]))
            # Get ROI
            stop_line_box = cv2.selectROI('Select box', frame, False, False)
            stop_line_box = [
                stop_line_box[1], 
                stop_line_box[0],
                stop_line_box[3] + stop_line_box[1],
                stop_line_box[2] + stop_line_box[0],
                ]
            cv2.destroyWindow('Select box') 
            print(stop_line_box)

        if frameId % multiplier == 0:
            # Detect persons
            image_np = frame.copy()
            w_image, h_image, _ = image_np.shape
            
            if USE_TF_SERVING:
                payload = {"instances": [image_np.tolist()]}
                res = requests.post(tf_serving_url, json=payload)
                detections = res.json()["predictions"][0]
            else:
                input_tensor = tf.convert_to_tensor(image_np)
                input_tensor = input_tensor[tf.newaxis, ...]
                detections = detection_model(input_tensor)
            
            # Prepare response
            scores = np.array(detections['detection_scores']).squeeze()
            labels = np.array(detections['detection_classes']).squeeze()

            boxes = np.array(detections['detection_boxes']).squeeze()
            boxes = np.array(boxes)
            boxes = boxes*np.array([w_image, h_image, w_image, h_image])
            boxes = boxes.astype('int16')

            # Visualize results
            res_image = image_np.copy()
            
            for i in range(len(scores)):
                if scores[i] > score_tresh:
                    box = boxes[i] 
                    box_color = (0,255,0)
                    # TODO: ALARM! Update base!
                    if  bb_intersection_over_union(box, stop_line_box) > iou_thresh:
                        # print(bb_intersection_over_union(box, stop_line_box))
                        box_color = (0,0,255)
                        alarm_time_on_video.append(str(datetime.timedelta(seconds=int(seconds*frameId/multiplier))))
                        print('-->',str(datetime.timedelta(seconds=int(frameId/multiplier))))
                    
                    # print box
                    cv2.rectangle(res_image, (box[1], box[0]), (box[3], box[2]), box_color)

            cv2.rectangle( 
                res_image, 
                (stop_line_box[1], stop_line_box[0]), 
                (stop_line_box[3], stop_line_box[2]), 
                (0,0,255))

            cv2.imshow("Frame", res_image)
            if WRITE_RESULT_VIDEO:
                video_writer.write(res_image)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        print('\r{}/{}'.format(frameId,frame_count),end='')

    # Do a bit of cleanup
    vs.release()
    if WRITE_RESULT_VIDEO:
        video_writer.release()
    cv2.destroyAllWindows()

    # Update db
    result_json = {
        'filename':video_filename,
        'w_h':[w_image, h_image],
        'alarm_box':stop_line_box,
        'alarm_time':alarm_time_on_video
    }

    result_json_filename = '{}_db.json'.format(video_filename)
    with open(os.path.join(data_base_path, result_json_filename), 'w') as f:
        json.dump(result_json, f)

print("[INFO] Done!")