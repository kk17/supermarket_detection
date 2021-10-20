import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.core import dataset as D
from fiftyone import ViewField as F
import tensorflow as tf
import os
from PIL import Image
from six import BytesIO
import numpy as np
from supermarket_detection import model_utils


def build_dataset(classes,
                  source_dataset='open-images-v6',
                  split="validation",
                  label_types=['detections'],
                  max_samples_per_class=10,
                  delete_exist_dataset=False):

    all_calss_dataset_name = f'{source_dataset}-{split}-{max_samples_per_class}'
    dataset = None
    if D.dataset_exists(all_calss_dataset_name):
        if delete_exist_dataset:
            print(f'Delete existing dataset {all_calss_dataset_name}')
            D.delete_dataset(all_calss_dataset_name)
        else:
            print(f'Load existing dataset {all_calss_dataset_name}')
            dataset = D.load_dataset(all_calss_dataset_name)
    if dataset:
        return dataset

    single_class_datasets = []
    for c in classes:
        dataset_name = f'{source_dataset}-{split}-{c}-{max_samples_per_class}'
        single_class_dataset = None
        if D.dataset_exists(dataset_name):
            if delete_exist_dataset:
                print(f'Delete existing dataset {dataset_name}')
                D.delete_dataset(dataset_name)
            else:
                print(f'Load existing dataset {dataset_name}')
                single_class_dataset = D.load_dataset(dataset_name)

        if not single_class_dataset:
            single_class_dataset = foz.load_zoo_dataset(
                source_dataset,
                split=split,
                label_types=label_types,
                classes=[c],
                max_samples=max_samples_per_class,
                dataset_name=dataset_name)
        single_class_datasets.append(single_class_dataset)

    dataset = single_class_datasets[0].clone(all_calss_dataset_name)
    for d in single_class_datasets[1:]:
        dataset.merge_samples(d)
    return dataset


# Bboxes are in [top-left-x, top-left-y, width, height] format
bbox_area = F("bounding_box")[2] * F("bounding_box")[3]


def build_dateset_view(dataset,
                       dataset_type='open_image',
                       IsOccluded=None,
                       IsTruncated=None,
                       IsGroupOf=None,
                       IsDepiction=None,
                       IsInside=None,
                       iscrowd=None,
                       valid_labels=None,
                       bbox_area_lower_bound=None):
    if dataset_type == 'open_image':
        gt_field = 'detections'
    else:
        gt_field = 'ground_truth'
    view = dataset
    if valid_labels:
        view = view.filter_labels(gt_field,
                                  F("label").is_in(valid_labels),
                                  only_matches=False)
    if IsOccluded is not None:
        view = view.filter_labels(gt_field,
                                  F("IsOccluded") == IsOccluded,
                                  only_matches=True)
    if IsTruncated is not None:
        view = view.filter_labels(gt_field,
                                  F("IsTruncated") == IsTruncated,
                                  only_matches=True)
    if IsGroupOf is not None:
        view = view.filter_labels(gt_field,
                                  F("IsGroupOf") == IsGroupOf,
                                  only_matches=True)
    if IsDepiction is not None:
        view = view.filter_labels(gt_field,
                                  F("IsDepiction") == IsDepiction,
                                  only_matches=True)
    if IsInside is not None:
        view = view.filter_labels(gt_field,
                                  F("IsInside") == IsInside,
                                  only_matches=True)
    if iscrowd is not None:
        view = view.filter_labels(gt_field,
                                  F("iscrowd") == iscrowd,
                                  only_matches=True)
    if bbox_area_lower_bound:
        view = view.filter_labels(
            gt_field,
            bbox_area > bbox_area_lower_bound,
        )
    return view


def export_dataset_or_view(
    dataset_or_view,
    export_dir,
    label_field,
    dataset_type=fo.types.COCODetectionDataset,
    overwrite=False,
    tf_file_name=None,
):
    dataset_or_view.export(export_dir=export_dir,
                           dataset_type=dataset_type,
                           label_field=label_field,
                           overwrite=overwrite)

    if dataset_type == fo.types.TFObjectDetectionDataset and tf_file_name:
        if not tf_file_name.endswith('.records'):
            tf_file_name += '.records'
        source = os.path.join(export_dir, 'tf.records')
        target = os.path.join(export_dir, tf_file_name)
        if os.path.exists(target) and overwrite:
            os.remove(target)
        os.rename(source, target)


def check_tf_record(filepath, take_num=1):
    raw_dataset = tf.data.TFRecordDataset(filepath)
    for raw_record in raw_dataset.take(take_num):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)
        # print('-' * 32)
        # n = len(example.features)
        # for i in range(n):
        #     feature = example.features[i]
        #     if feature.key != 'image/encoded':
        #         print(feature)


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def add_prediction_for_fiftyone_dataset(dataset_or_view,
                                        detect_fn,
                                        category_index=None,
                                        classes=None,
                                        confidence_threshold=None):
    label_id_offset = 1
    if not classes and category_index:
        classes = {k: v['name'] for k, v in category_index.items()}

    # Add predictions to samples
    with fo.ProgressBar() as pb:
        for sample in pb(dataset_or_view):
            image_np = load_image_into_numpy_array(sample.filepath)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0),
                                                dtype=tf.float32)

            # Perform inference
            preds = detect_fn(input_tensor)
            class_ids = preds["detection_classes"][0].numpy().astype(
                np.uint32) + label_id_offset
            scores = preds["detection_scores"][0].numpy()
            boxes = preds["detection_boxes"][0].numpy()

            # Convert detections to FiftyOne format
            detections = []
            for class_id, score, box in zip(class_ids, scores, boxes):
                if confidence_threshold and score < confidence_threshold:
                    continue

                detections.append(
                    fo.Detection(label=classes[class_id],
                                 bounding_box=box.tolist(),
                                 confidence=score))

            # Save predictions to dataset
            sample["predictions"] = fo.Detections(detections=detections)
            sample.save()


def load_tfrecord_as_fiftyone_dataset(dataset_dir,
                                      tf_records_path=None,
                                      images_dir=None):
    # dataset_dir (None) – the dataset directory
    # images_dir (None) – the directory in which the images will be written. If not provided, the images will be unpacked into dataset_dir
    # tf_records_path (None) –
    # an optional parameter that enables explicit control over the location of the TF records. Can be any of the following:
    # a filename like "tf.records" or glob pattern like "*.records-*-of-*" specifying the location of the records in dataset_dir
    # an absolute filepath or glob pattern for the records. In this case, dataset_dir has no effect on the location of the records
    # If None, the parameter will default to *record*
    # image_format (None) – the image format to use to write the images to disk. By default, fiftyone.config.default_image_ext is used
    # max_samples (None) – a maximum number of samples to import. By default, all samples are imported
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        images_dir=images_dir,
        tf_records_path=tf_records_path,
        dataset_type=fo.types.TFObjectDetectionDataset,
    )
    return dataset