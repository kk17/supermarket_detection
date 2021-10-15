import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder


def load_model_from_checkpoint(config_path, checkpoint_path):
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(config_path)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(checkpoint_path, 'ckpt-0')).expect_partial()
    return detection_model

@tf.function
def detect_fn(detection_model, image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

def create_category_index(label_filepath):
    category_index = label_map_util.create_category_index_from_labelmap(label_filepath,
        use_display_name=True)
    return category_index