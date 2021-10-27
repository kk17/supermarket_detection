import yaml
class ObjectDetection:
    def __init__(self,
                 load_model_from_checkpoint=False,
                 model_dir=None,
                 checkpoint_dir=None,
                 pipeline_config_path=None,
                 checkpoint_no=0,
                 label_map_path=None,
                 min_score_thresh=0.3,
                 detect_every_n_frame=80):
        self.load_model_from_checkpoint = load_model_from_checkpoint
        self.model_dir = model_dir
        self.pipeline_config_path = pipeline_config_path
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_no = checkpoint_no
        self.label_map_path = label_map_path
        self.min_score_thresh = min_score_thresh
        self.detect_every_n_frame = detect_every_n_frame


class Config:
    def __init__(self, object_detection=ObjectDetection):
        self.object_detection = object_detection


def load_from_yaml(filepath):
    with open(filepath,'r') as f:
        cfg_dict = yaml.full_load(f)
    if 'object_detection' in cfg_dict:
        cfg_dict['object_detection'] = ObjectDetection(**cfg_dict['object_detection'])
    return Config(**cfg_dict)