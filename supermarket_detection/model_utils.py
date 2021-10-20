import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
from functools import partial


def load_saved_model(model_path):
    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(model_path)
    return detect_fn


def load_model_from_checkpoint(config_path,
                               checkpoint_path,
                               checkpoint_no=0,
                               as_detect_fn=True,
                               preprocess_in_graph=False):
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config,
                                          is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(checkpoint_path,
                              f'ckpt-{checkpoint_no}')).expect_partial()
    if as_detect_fn:
        if preprocess_in_graph:
            detect_fn_partial = partial(detect_fn, detection_model=detection_model)
            detect_fn_partial_graph = tf.function(detect_fn_partial)
            return detect_fn_partial_graph
        else:
            predict_and_postprocess_partial = partial(_predict_and_postprocess, detection_model=detection_model)
            predict_and_postprocess_partial_graph = tf.function(predict_and_postprocess_partial)
            def _detect_fn(image):
                image, shapes = detection_model.preprocess(image)
                return predict_and_postprocess_partial_graph(image=image, shapes=shapes)
            return _detect_fn
    else:
        return detection_model


def detect_fn(detection_model, image):
    """Detect objects in image."""
    # preprocess cannot wrap into tf function, because image may have different shape before preprocess
    image, shapes = detection_model.preprocess(image)
    return _predict_and_postprocess(detection_model, image, shapes)


def _predict_and_postprocess(detection_model, image, shapes):
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def create_category_index(label_filepath):
    category_index = label_map_util.create_category_index_from_labelmap(
        label_filepath, use_display_name=True)
    return category_index