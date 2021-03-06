#!/usr/bin/env python

from numpy.lib.shape_base import take_along_axis
from object_detection.protos import post_processing_pb2
import pandas as pd
import numpy as np
import cv2
import stopwatch
# from supermarket_detection.dataset_utils import load_image_into_numpy_array
from supermarket_detection import model_utils, config, detection_utils, image_utils
import tensorflow as tf
import os
from object_detection.utils import visualization_utils as viz_utils
import argparse
import logging
import re
from skimage import io
from stopwatch import Stopwatch


def load_model_and_category_index(cfg):
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
    return detection_model, category_index


def detect_from_image_numpy(detection_model, image_np):
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
        logging.info(
            "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".
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
            detections = detect_from_image_numpy(detection_model, image_np)
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


def merge_bounding_box_for_one_class(class_name_to_iou_map, category_index,
                                   boxes, classes, scores):
    n = len(boxes)
    class_names = [category_index[c]['name'] for c in classes]
    name_to_ids = {category_index[c]['name']: c for c in classes}
    for _cn, merge_min_iou_thresh in class_name_to_iou_map.items():
        rest_boxes, rest_class_names, rest_scores = [], [], []
        _boxes, _class_names, _scores = [], [], []
        for b, cn, s in zip(boxes, class_names, scores):
            if cn == _cn:
                _boxes.append(b)
                _class_names.append(cn)
                _scores.append(s)
            else:
                rest_boxes.append(b)
                rest_class_names.append(cn)
                rest_scores.append(s)
        _boxes, _class_names, _scores = detection_utils.merge_bounding_boxes(
            _boxes, _class_names, _scores, merge_min_iou_thresh)
        boxes = rest_boxes + _boxes
        class_names = rest_class_names + _class_names
        scores = rest_scores + _scores
        classes = [name_to_ids[cn] for cn in class_names]
        _n = len(boxes)
        if n - _n > 0:
            logging.info(f'Reduced {n - _n} boxes for class {_cn} in post processing')
    return np.asarray(boxes), np.asarray(classes), np.asarray(scores)


def remove_high_iou_bounding_boxes(min_iou_thresh, class_weight_order, category_index,
                                   boxes, classes, scores, round_ndigits=2):
    logging.debug(f"min_iou_thresh: {min_iou_thresh}")
    n = len(boxes)
    class_names = [category_index[c]['name'] for c in classes]
    name_to_ids = {category_index[c]['name']: c for c in category_index.keys()}
    id_to_weights = {name_to_ids[c]: w for w, c in enumerate(class_weight_order)}
    removed = [False] * n
    for i in range(n):
        if removed[i]:
            continue
        for j in range(i+1, n):
            if removed[j]:
                continue
            iou = detection_utils.bb_intersection_over_union(boxes[i], boxes[j])
            logging.debug(f'Class {class_names[i]} box{i}: {boxes[i]} {class_names[j]} box{j}:{boxes[j]} iou: {iou}')
            if iou >= min_iou_thresh:
                remove_box_index = i
                si, sj = round(scores[i], round_ndigits), round(scores[i], round_ndigits)
                logging.debug(f'{scores[i]} {scores[j]}, {si} {sj}')
                if si > sj or (si == sj and id_to_weights[classes[i]] > id_to_weights[classes[j]]):
                    remove_box_index = j
                removed[remove_box_index] = True
    boxes = [boxes[i] for i in range(n) if not removed[i]]
    classes = [classes[i] for i in range(n) if not removed[i]]
    scores = [scores[i] for i in range(n) if not removed[i]]
    _n = len(boxes)
    if n - _n > 0:
        logging.info(f'Reduced {n - _n} boxes for all classes in post processing')
    return np.asarray(boxes), np.asarray(classes), np.asarray(scores)

def remove_too_small_bounding_boxes(min_area_thresh, category_index, boxes, classes, scores):
    logging.debug(f"min_area_thresh: {min_area_thresh}")
    n = len(boxes)
    removed = [False] * n
    for i, b in enumerate(boxes):
        area = (b[2] - b[0]) * (b[3] - b[1])
        logging.debug(f'box{i} area: {area}')
        if area < min_area_thresh:
            removed[i] = True
            logging.info(f'Removed too small box {category_index[classes[i]]}, area {area} < min_area_thresh: {min_area_thresh}')
    boxes = [boxes[i] for i in range(n) if not removed[i]]
    classes = [classes[i] for i in range(n) if not removed[i]]
    scores = [scores[i] for i in range(n) if not removed[i]]
    _n = len(boxes)
    if n - _n > 0:
        logging.info(f'Removed {n - _n} too small boxes in post processing')
    return np.asarray(boxes), np.asarray(classes), np.asarray(scores)


def use_classifer_for_classes(classifer_model, classname_to_id_map, image_np,
                              category_index, boxes, classes, scores):
    detect_name_to_ids = {category_index[c]['name']: c for c in category_index.keys()}
    classifer_id_to_detect_id_map = {
        cid: detect_name_to_ids[c]
        for c, cid in classname_to_id_map.items()
    }
    detect_ids = set(
        [detect_name_to_ids[cn] for cn in classname_to_id_map.keys()])
    logging.debug(f'{detect_name_to_ids}: detect_name_to_ids')
    logging.debug(f'{classifer_id_to_detect_id_map}: classifer_id_to_detect_id_map')
    logging.debug(f'{detect_ids}: detect_ids')

    indexes = []
    images = []
    for i, (b, c, s) in enumerate(zip(boxes, classes, scores)):
        if c in detect_ids:
            logging.debug(f'c: {c}')
            indexes.append(i)
            bb_image_np = image_utils.crop_by_bounding_box(image_np, b)
            images.append(bb_image_np)

    if images:
        c_classes, _scores = model_utils.make_prediction(images, classifer_model)
        for i, (cc, _s) in enumerate(zip(c_classes, _scores)):
            index = indexes[i]
            _c = classifer_id_to_detect_id_map[cc]
            logging.debug(f'_c: {_c}')
            c = classes[index]
            s = scores[index]
            if _s > s and _c != c:
                cn, _cn = category_index[c]['name'], category_index[_c]['name']
                logging.info(
                    f'Replace box {index} class {cn} to {_cn} from classifier result')
                classes[index] = _c
                scores[index] = _s


def detect_from_directory(cfg,
                          detection_model,
                          category_index,
                          pred_df,
                          export_images,
                          inputpath,
                          outputpath,
                          class_name_to_csv_header_mapping,
                          min_score_thresh=0.3,
                          classifer_model=None):

    stopwatch_all = Stopwatch()
    stopwatch_all.start()
    label_id_offset = 1

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    if os.path.isfile(inputpath):
        filenames = [os.path.basename(inputpath)]
        inputpath = os.path.dirname(inputpath)
    else:
        filenames = os.listdir(inputpath)
    image_filename_pattern = re.compile('.+\.(png|jpg)$', re.IGNORECASE)

    sw_image = Stopwatch()
    sw_step = Stopwatch()
    for filename in filenames:
        if not image_filename_pattern.search(filename):
            continue
        filepath = f'{inputpath}/{filename}'
        sw_image.restart()
        sw_step.restart()
        logging.info(f'Loading image file: {filepath}')
        # image_np = load_image_into_numpy_array(filepath)
        image_np = io.imread(filepath)
        logging.info(f'Loaded image file, time: {sw_step}')
        if len(image_np.shape) < 3:
            continue
        logging.info(f'Start detection for file: {filepath}')
        sw_step.restart()
        detections = detect_from_image_numpy(detection_model, image_np)
        logging.info(f'Finished dection, time: {sw_step}')
        if detections:
            boxes = detections['detection_boxes'][0].numpy()
            classes = (detections['detection_classes'][0].numpy() +
                       label_id_offset).astype(int)
            scores = detections['detection_scores'][0].numpy()

            if cfg.post_processing.use_classifer_for_classes:
                sw_step.restart()
                logging.info(f'Do additional classification')
                use_classifer_for_classes(
                    classifer_model,
                    cfg.post_processing.use_classifer_for_classes, image_np,
                    category_index, boxes, classes, scores)
                logging.info(f'Finish additional classification, time: {sw_step}')

            if cfg.post_processing.merge_bounding_box_for_one_class:
                sw_step.restart()
                logging.info(f'Merge bounding boxes')
                boxes, classes, scores = merge_bounding_box_for_one_class(
                    cfg.post_processing.merge_bounding_box_for_one_class,
                    category_index, boxes, classes, scores)
                logging.info(f'Merged bounding boxes, time: {sw_step}')

            if cfg.post_processing.remove_high_iou_bounding_boxes:
                sw_step.restart()
                logging.info(f'Remove high iou bounding boxes')
                boxes, classes, scores = remove_high_iou_bounding_boxes(
                    cfg.post_processing.remove_high_iou_bounding_boxes.min_iou_thresh,
                    cfg.post_processing.remove_high_iou_bounding_boxes.class_weight_order,
                    category_index, boxes, classes, scores)
                logging.info(f'Removed high iou bounding boxes, time: {sw_step}')
            
            if cfg.post_processing.remove_too_small_bounding_box_min_area:
                sw_step.restart()
                logging.info(f'Remove too small bounding boxes')
                boxes, classes, scores = remove_too_small_bounding_boxes(
                    cfg.post_processing.remove_too_small_bounding_box_min_area, category_index, boxes, classes, scores)
                logging.info(f'Removed too small bounding boxes, time: {sw_step}')

            #draw bounding box
            if export_images:
                logging.info(f'Visualize boxes')
                sw_step.restart()
                viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=10,
                    min_score_thresh=min_score_thresh,
                    agnostic_mode=False)
                logging.info(f'Visualized boxes, time: {sw_step}')
                logging.info(f'Export image to {outputpath}/{filename}')
                sw_step.restart()
                # viz_utils.save_image_array_as_png(image_np,
                #                                   f'{outputpath}/{filename}')
                io.imsave(f'{outputpath}/{filename}', image_np)
                logging.info(f'Exported image, time: {sw_step}')
                # im_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'{outputpath}/{filename}', im_bgr)

            #count items
            item_count = {}
            item_count['Id'] = re.findall(r'(.*)(?:\.)', filename)[0]
            logging.info(item_count['Id'])
            logging.info(f'Start counting')
            sw_step.restart()
            for i in range(scores.shape[0]):
                class_name = category_index[classes[i]]['name']
                if scores[i] >= min_score_thresh:
                    if class_name_to_csv_header_mapping:
                        header = class_name_to_csv_header_mapping[
                            class_name]
                    else:
                        header = class_name
                    try:
                        item_count[header] += 1
                    except:
                        item_count[header] = 1
                else:
                    logging.info(f'Not counting {class_name} {i}, score {scores[i]} < min_score_thresh {min_score_thresh}')

            pred_df = pred_df.append(item_count, ignore_index=True)
            logging.info(f'Finished counting, time: {sw_step}')
            logging.info(
                f'result:\n{pred_df.iloc[-1,:]}\nused time: {sw_image}')
    logging.info(f"Completed predictions. Total used time: {stopwatch_all}")
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
    parser.add_argument("--export_images", "-e", action="store_true")
    parser.add_argument("--camera", "-c", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')


    cfg = config.load_from_yaml(args.config).object_detection
    stopwatch = Stopwatch()

    logging.info("Loading model")
    stopwatch.start()
    model, catagory = load_model_and_category_index(cfg)
    if cfg.class_name_to_csv_header_mapping:
        headers = ['Id'] + [
            header for header in cfg.class_name_to_csv_header_mapping.values()
        ]
    else:
        headers = ['Id'
                   ] + [catagory[index]['name'] for index in catagory.keys()]
    logging.info(f"Loaded model, time: {stopwatch}")
    logging.info('Initiating  object detection model')
    stopwatch.restart()
    # make a detection using a fake image to initiate the model
    fake_iamge_np = np.zeros((100, 100, 3))
    detect_from_image_numpy(model, fake_iamge_np)
    logging.info(f'Model initiated, time: {stopwatch}')
    stopwatch.stop()

    classifer_model = None
    if cfg.post_processing.use_classifer_for_classes:
        stopwatch.restart()
        logging.info('Load and init classifer detection model')
        classifer_model = model_utils.load_classificaton_model(
            cfg.post_processing.classifier_model_path)
        classifer_model.predict(np.ones((1, 224, 224, 3)))
        logging.info(f"Loaded model, time: {stopwatch}")
        stopwatch.stop()
    if args.camera:
        detect_from_camera(model,
                           catagory,
                           min_score_thresh=cfg.min_score_thresh,
                           detect_every_n_frame=cfg.detect_every_n_frame)
    else:
        pred_df = pd.DataFrame(columns=headers)
        pred_df = detect_from_directory(cfg,
                                        model,
                                        catagory,
                                        pred_df,
                                        args.export_images,
                                        inputpath=args.inputpath,
                                        outputpath=args.outputpath,
                                        min_score_thresh=cfg.min_score_thresh,
                                        class_name_to_csv_header_mapping=cfg.
                                        class_name_to_csv_header_mapping,
                                        classifer_model=classifer_model)
        try:
            pred_df['Id'] = pred_df['Id'].astype(int)
        except:
            pass
        pred_df = pred_df.fillna(0).sort_values('Id')
        pred_df.to_csv(f'{args.outputpath}/pred_df.csv',
                       index=0,
                       errors='ignore')


if __name__ == '__main__':
    # python - import side effects on logging: how to reset the logging module? - Stack Overflow
    # https://stackoverflow.com/questions/12034393/import-side-effects-on-logging-how-to-reset-the-logging-module
    # root = logging.getLogger()
    # list(map(root.removeHandler, root.handlers))
    # list(map(root.removeFilter, root.filters))
    main()
