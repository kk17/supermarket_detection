#!/usr/bin/env python

from object_detection.utils import config_util
import os


def update_pipeline_config_file(
    pipeline_config,
    origin_pipeline_config=None,
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
    max_number_of_boxes=25,
    max_detections_per_class=5,
    max_total_detections=5
):
    if not origin_pipeline_config:
        origin_pipeline_config = pipeline_config
    configs = config_util.get_configs_from_pipeline_file(origin_pipeline_config)
    model_config = configs['model']
    model = None
    if model_config.faster_rcnn.num_classes != 0:
        model = model_config.faster_rcnn
    elif model_config.ssd.num_classes != 0:
        model = model_config.ssd
        model_config.ssd.freeze_batchnorm = True
    elif model_config.center_net.num_classes != 0:
        model = model_config.center_net
    elif model_config.experimental_model.num_classes != 0:
        model = model_config.experimental_model
    model.num_classes = num_classes
    if max_detections_per_class:
        model.post_processing.batch_non_max_suppression.max_detections_per_class = max_detections_per_class
    if max_total_detections:
        model.post_processing.batch_non_max_suppression.max_total_detections = max_total_detections


    train_config = configs['train_config']
    if batch_size:
        train_config.batch_size = batch_size
    train_config.fine_tune_checkpoint = fine_tune_checkpoint
    train_config.fine_tune_checkpoint_type = fine_tune_checkpoint_type
    train_config.fine_tune_checkpoint_version = fine_tune_checkpoint_version
    train_config.use_bfloat16 = use_bfloat16
    if max_number_of_boxes:
        train_config.max_number_of_boxes = max_number_of_boxes

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
    from dotenv import load_dotenv
    import argparse

    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    
    load_dotenv()

    model_name = os.environ.get('MODEL_NAME', '')
    model_version = os.environ.get('MODEL_VERSION', 'v1')
    checkpoint_num = os.environ.get('CHECKPOINT_NUM', '0')
    dataset_dir = os.environ.get('DATASET_DIR', 'data/custom01')

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", '-n', type=str, dest='model_name', default=model_name,
                        help="model name")
    parser.add_argument("--version", '-v', type=str, dest='model_version', default=model_version,
                        help="model version")
    parser.add_argument("--checkpoint-num", '-cn', type=int, default=int(checkpoint_num),
                        help="checkpoint num")
    parser.add_argument("--dataset-dir", '-d', type=str, default=dataset_dir,
                        help="dataset dir")
    parser.add_argument("--copy-from-version", '-fv', type=str,
                        help="copy config from model version")
    parser.add_argument("--copy-from-pre-trained", '-fp', action='store_true',
                        help="copy config from model version")

    args = parser.parse_args()
    print(args)

    if not args.model_name:
        print('model name must be specified')
        return


    workspace_dir = os.path.abspath(os.path.dirname(__file__))
    pipeline_config = os.path.join(workspace_dir, 'models', args.model_name, args.model_version, 'pipeline.config')
    origin_pipeline_config = pipeline_config
    if args.copy_from_version:
        origin_pipeline_config = os.path.join(workspace_dir, 'models', args.model_name, args.copy_from_version, 'pipeline.config')
    elif args.copy_from_pre_trained:
        origin_pipeline_config = os.path.join(workspace_dir, 'pre_trained_models', args.model_name, 'pipeline.config')

    fine_tune_checkpoint = os.path.join('pre_trained_models', args.model_name, 'checkpoint', f'ckpt-{args.checkpoint_num}')
    train_input_path = os.path.join(args.dataset_dir, "train.tfrecord")
    label_map_path = os.path.join(args.dataset_dir, "label_map.pbtxt")
    eval_input_path = os.path.join(args.dataset_dir, "valid.tfrecord")

    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    import sys
    sys.path.append('..')
    from supermarket_detection import model_utils

    category_index = model_utils.create_category_index(label_map_path)
    num_classes = len(category_index)

    print(f'origin_pipeline_config: {origin_pipeline_config}')
    print(f'pipeline_config: {pipeline_config}')
    update_config = {
        'num_classes': num_classes,
        'fine_tune_checkpoint': fine_tune_checkpoint,
        'train_input_path': train_input_path,
        'train_label_map_path': label_map_path,
        'eval_input_path': eval_input_path,
        'eval_label_map_path': label_map_path,
        'batch_size': 4,
    }

    print(f'update_config: {update_config}')
    update_pipeline_config_file(pipeline_config, origin_pipeline_config=origin_pipeline_config, **update_config)
    print('Finished')


if __name__ == '__main__':
  main()