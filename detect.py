#!/usr/bin/env python

import pandas as pd
import numpy as np
import cv2
from supermarket_detection.dataset_utils import load_image_into_numpy_array
from supermarket_detection import model_utils, config
import tensorflow as tf
import os
from object_detection.utils import visualization_utils as viz_utils
import argparse
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def load_model_and_category_index(cfg):
    logging.info("Loading model")
    if cfg.load_model_from_checkpoint:
        detection_model = model_utils.load_model_from_checkpoint(
            cfg.pipeline_config_path,
            cfg.checkpoint_dir,
            checkpoint_no=cfg.checkpoint_no,
            as_detect_fn=True)
        detection_model.load_model_from_checkpoint = True
    else:
        detection_model = model_utils.load_saved_model(cfg.model_dir)
        detection_model.load_model_from_checkpoint = False
    category_index = model_utils.create_category_index(cfg.label_map_path)
    logging.info("Loaded model")
    return detection_model, category_index


def detect_from_image_numpy(detection_model, category_index, min_score_thresh,
                            image_np):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    if detection_model.load_model_from_checkpoint:
        input_tensor = tf.convert_to_tensor(image_np_expanded,
                                            dtype=tf.float32)
    else:
        input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.uint8)
    detections = detection_model(input_tensor)
    return detections


def detect_from_camera(detection_model,
                       category_index,
                       min_score_thresh=0.3,
                       detect_every_n_frame=80,
                       vedio_file=0):
    cap = cv2.VideoCapture(vedio_file)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        logging.info("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".
              format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(
            "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(
                fps))

    count = 0
    shape = None
    label_id_offset = 1
    detections = None
    while True:
        # Read frame from camera
        ret, image_np = cap.read()
        if not shape:
            shape = image_np.shape
            logging.info(f'image shape: {shape}')

        if count % detect_every_n_frame == 0:
            logging.info(f'Start dectect frame: {count}')
            detections = detect_from_image_numpy(detection_model,
                                                 category_index,
                                                 min_score_thresh, image_np)
            # Display output
            logging.info(f'Finished dectect frame: {count}')

        if detections:
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() +
                 label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=10,
                min_score_thresh=min_score_thresh,
                agnostic_mode=False)

        cv2.imshow('object detection', image_np)
        count += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_from_directory(detection_model,
                        category_index,
                        pred_df,
                        export_images,
                        inputpath,
                        outputpath,
                        min_score_thresh=0.3):
    
    label_id_offset = 1
    
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        
    filenames = os.listdir(inputpath)
    
    for filename in filenames:
        image_np = load_image_into_numpy_array(f'{inputpath}/{filename}')
        
        detections = detect_from_image_numpy(detection_model,
                                            category_index,
                                            min_score_thresh,
                                            image_np)
        
        if detections:
            #draw bounding box 
            if export_images:
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    detections['detection_boxes'][0].numpy(),
                    (detections['detection_classes'][0].numpy() +
                    label_id_offset).astype(int),
                    detections['detection_scores'][0].numpy(),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=10,
                    min_score_thresh=min_score_thresh,
                    agnostic_mode=False)
                viz_utils.save_image_array_as_png(image_np, f'{outputpath}/{filename}')
                # im_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'{outputpath}/{filename}', im_bgr)  

            #count items
            item_count = {}
            item_count['Id'] = re.findall(r'(.*)(?:\.)',filename)[0]
            logging.info(item_count['Id'])
            
            scores = detections['detection_scores'][0]
            classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
            
            for i in range(scores.shape[0]):
                if scores is None or scores[i] > min_score_thresh:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                        try:
                            item_count[class_name] += 1 
                        except:
                            item_count[class_name] = 1 
                            
            pred_df = pred_df.append(item_count, ignore_index=True) 
            logging.info(pred_df.iloc[-1,:])     
    logging.info("Completed predictions")
    return pred_df

def main():

    TF_CPP_MIN_LOG_LEVEL = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL
    tf.get_logger().setLevel('ERROR')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        "-f",
                        type=str,
                        default="config/default.yml")
    parser.add_argument("--inputpath", 
                        "-i",
                        type=str,
                        default='workspace/data/test')
    parser.add_argument("--outputpath",
                        "-o",
                        type=str,
                        default='workspace/output/test')
    parser.add_argument("--export_images",
                        "-e",
                        action="store_true")
    parser.add_argument("--camera",
                        "-c",
                        action="store_true")
    args = parser.parse_args()
    cfg = config.load_from_yaml(args.config).object_detection
    model, catagory = load_model_and_category_index(cfg)
    class_names = [catagory[index]['name']  for index in catagory.keys()] 
    if args.camera:
        detect_from_camera(model,
                        catagory,
                        min_score_thresh=cfg.min_score_thresh,
                        detect_every_n_frame=cfg.detect_every_n_frame)
    else: 
        pred_df = pd.DataFrame(columns=['Id'] + class_names)  
        pred_df = detect_from_directory(model,
                        catagory,
                        pred_df,
                        args.export_images,
                        inputpath=args.inputpath,
                        outputpath=args.outputpath,
                        min_score_thresh=cfg.min_score_thresh)
        try:
            pred_df['Id'] = pred_df['Id'].astype(int)
        except:
            pass
        pred_df = pred_df.fillna(0).sort_values('Id')
        pred_df.to_csv(f'{args.outputpath}/pred_df.csv', index=0, errors='ignore')

if __name__ == '__main__':
    main()
