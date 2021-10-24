#!/usr/bin/env python

import numpy as np
import cv2
from supermarket_detection import model_utils, config
import tensorflow as tf
import os
from object_detection.utils import visualization_utils as viz_utils
import argparse


def load_model_and_category_index(cfg):
    print("Loading model")
    if cfg.load_model_from_checkpoint:
        detection_model = model_utils.load_model_from_checkpoint(cfg.pipeline_config_path, cfg.checkpoint_dir, checkpoint_no=cfg.checkpoint_no, as_detect_fn=True)
        detection_model.load_model_from_checkpoint = True
    else:
        detection_model = model_utils.load_saved_model(cfg.model_dir) 
        detection_model.load_model_from_checkpoint = False
    category_index = model_utils.create_category_index(cfg.label_map_path)
    print("Loaded model")
    return detection_model, category_index


def detect_from_camera(detection_model, category_index, min_score_thresh):
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame from camera
        ret, image_np = cap.read()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        if detection_model.load_model_from_checkpoint:
            input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.float32)
        else:
            input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.uint8)
        detections = detection_model(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'][0].numpy(),
              (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
              detections['detection_scores'][0].numpy(),
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=10,
              min_score_thresh=min_score_thresh,
              agnostic_mode=False)

        # Display output
        cv2.imshow('object detection', image_np_with_detections)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():

    TF_CPP_MIN_LOG_LEVEL = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CPP_MIN_LOG_LEVEL
    tf.get_logger().setLevel('ERROR') 

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-f", type=str, default="config/default.yml")
    args = parser.parse_args()
    cfg = config.load_from_yaml(args.config).object_detection
    model, catagory = load_model_and_category_index(cfg)
    detect_from_camera(model, catagory, cfg.min_score_thresh)


if __name__ == '__main__':
  main()