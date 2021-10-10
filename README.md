# Supermarket detection

A project combine object dectection and face recognition.

## Init workspace

Install python packages and prepare workspace

```bash
./init.sh
```

## Build a dataset for object dections

Use fiftyone to get the require class from open image v6 dataset and build a TFOjbectDetectionDataset. Check notebooks: [building_dataset.ipynb](https://github.com/kk17/supermark_det/blob/main/notebooks/building_dataset.ipynb)

## Train object detection model
Use TensorFlow Object Detection API to do transfer learning.

```bash
cd workspace
./pre_trained_models/download_model.sh
# TODO copy and modify pipeline.config
./train_model.sh
```

### Reference

- [How to Train Your Own Object Detector Using TensorFlow Object Detection API - neptune.ai](https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api)

## Export and evaluate model

```bash
cd workspace
./export_model.sh
```

check notebooks: [model_for_inference.ipynb](https://github.com/kk17/supermark_det/blob/main/notebooks/model_for_inference.ipynb)


### Reference
- [TensorFlow Object Detection API: Best Practices to Training, Evaluation & Deployment - neptune.ai](https://neptune.ai/blog/tensorflow-object-detection-api-best-practices-to-training-evaluation-deployment)
## Reference
### Object detection

- [interactive_eager_few_shot_od_training_colab.ipynb - Colaboratory](https://colab.research.google.com/github/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb)
- [models/using_your_own_dataset.md at master · tensorflow/models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)
- [fiftyone - evaluate_detections](https://colab.research.google.com/github/voxel51/fiftyone/blob/v0.13.3/docs/source/tutorials/evaluate_detections.ipynb)

### Face Recognition

- [R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X](https://github.com/R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X)