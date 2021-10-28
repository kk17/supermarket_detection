#!/usr/bin/env python

import numpy as np
import cv2
from supermarket_detection.dataset_utils import load_image_into_numpy_array
from supermarket_detection import model_utils, config
import tensorflow as tf
import os
from object_detection.utils import visualization_utils as viz_utils
import argparse
import logging

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

        im_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f'{outputpath}/{filename}', im_bgr)        
    logging.info("Completed predictions")

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
    parser.add_argument("--camera",
                        "-c",
                        action="store_true")
    args = parser.parse_args()
    cfg = config.load_from_yaml(args.config).object_detection
    model, catagory = load_model_and_category_index(cfg)
    
    if args.camera:
        detect_from_camera(model,
                        catagory,
                        min_score_thresh=cfg.min_score_thresh,
                        detect_every_n_frame=cfg.detect_every_n_frame)
    else:    
        detect_from_directory(model,
                        catagory,
                        inputpath=args.inputpath,
                        outputpath=args.outputpath,
                        min_score_thresh=cfg.min_score_thresh)

if __name__ == '__main__':
    main()
