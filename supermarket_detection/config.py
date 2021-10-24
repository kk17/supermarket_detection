import yaml

class ObjectDetection(yaml.YAMLObject):
    yaml_tag = 'object_detection'
    def __init__(self):
        self.load_model_from_checkpoint = False
        self.model_dir = None
        self.label_map_path = None
        self.checkpoint_no = 0

class Config(yaml.YAMLObject):
    def __init__(self):
        self.object_detection = ObjectDetection()