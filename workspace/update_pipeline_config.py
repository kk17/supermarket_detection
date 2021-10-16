#!/usr/bin/env python

from object_detection.utils import config_util
import os


def update_pipeline_config_file(
    pipeline_config,
    num_classes=None,
    batch_size=None,
    fine_tune_checkpoint="",
    fine_tune_checkpoint_type="detection",
    fine_tune_checkpoint_version=2,
    train_input_path="",
    train_label_map_path="",
    use_bfloat16=False,
    eval_input_path="",
    eval_label_map_path="",
    eval_metrics_set="coco_detection_metrics",
    eval_use_moving_averages=False,
    
):
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    if model_config.faster_rcnn.num_classes != 0:
        model_config.faster_rcnn.num_classes = num_classes
    elif model_config.ssd.num_classes != 0:
        model_config.ssd.num_classes = num_classes
        model_config.ssd.freeze_batchnorm = True
    elif model_config.center_net.num_classes != 0:
        model_config.center_net.num_classes = num_classes
    elif model_config.experimental_model.num_classes != 0:
        model_config.experimental_model.num_classes = num_classes

    train_config = configs['train_config']
    if batch_size:
        train_config.batch_size = batch_size
    train_config.fine_tune_checkpoint = fine_tune_checkpoint
    train_config.fine_tune_checkpoint_type = fine_tune_checkpoint_type
    train_config.fine_tune_checkpoint_version = fine_tune_checkpoint_version
    train_config.use_bfloat16 = use_bfloat16

    train_input_reader = configs['train_input_config']
    train_input_reader.label_map_path = train_label_map_path
    train_input_reader.tf_record_input_reader.input_path[0] = train_input_path

    eval_config = configs['eval_config']
    eval_config.metrics_set[0] = eval_metrics_set
    eval_config.use_moving_averages = eval_use_moving_averages

    eval_input_reader = configs['eval_input_config']
    eval_input_reader.label_map_path = eval_label_map_path
    eval_input_reader.tf_record_input_reader.input_path[0] = eval_input_path
    
    output_dir = os.path.dirname(pipeline_config)
    config_util.save_pipeline_config(config_util.create_pipeline_proto_from_configs(configs), output_dir)

def main():
    model_name = os.environ.get('MODEL_NAME', '')
    model_version = os.environ.get('MODEL_VERSION', 'v1')
    checkpoint_num = os.environ.get('CHECKPOINT_NUM', '0')

    if not model_name:
        print('model_name must be specified')
        return

    workspace_dir = os.path.abspath(os.path.dirname(__file__))
    pipeline_config = os.path.join(workspace_dir, 'models', model_name, model_version, 'pipeline.config')
    fine_tune_checkpoint = os.path.join(workspace_dir, 'pre_trained_models', model_name, 'checkpoint', f'ckpt-{checkpoint_num}')

    print(f'pipeline_config: {pipeline_config}')
    args = {
        'num_classes': 4,
        'fine_tune_checkpoint': fine_tune_checkpoint,
        'train_input_path': "data/train.records",
        'train_label_map_path': "data/label_map.pbtxt",
        'eval_input_path': "data/evaluation.records",
        'eval_label_map_path': "data/label_map.pbtxt",
    }

    print(f'args: {args}')
    update_pipeline_config_file(pipeline_config, **args)
    print('Finished')


if __name__ == '__main__':
  main()