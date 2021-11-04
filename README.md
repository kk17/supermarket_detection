# Supermarket detection

A project combines object detection and faces recognition.
Object detection uses the Tensorflow Object Detection API.

## Project directory structure

```bash
├── build_dataset.py # script for build dataset using fiftyone
├── facenet # directory for scripts and files for object detection adapted from (https://github.com/R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X)
├── object_detect.py # script for loading a model and dectection
├── init.sh # script for install required packages 
├── notebooks # diretory for jupyter notebook
├── requirements.txt
├── supermarket_detection # python module diretory
├── tensorflow_model_garden # tensorflow model garden diretory, created by init.sh
└── workspace # workspace folder for training object detection models
    ├── data # dataset directory
    ├── exported_models # directory for exported models
    ├── models # directory for training models
    ├── pre_trained_models # directory for downloaded pre-trained model models
    ├── download_model.sh
    ├── select_model.sh
    ├── train_model.sh
    ├── evaluate_model.sh
    ├── export_model.sh
    ├── update_pipeline_config.py
    └── sync_workspace_with_drive.sh
```

## Train a object detection model

Note: You can run the jupyter notebook [notebooks/training_on_cloud.ipynb](./notebooks/training_on_cloud.ipynb) to prepare training on Google colab. After you open vscode in colab, the follow steps are the same.

### 0. install protobuf(for Windows system)

The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be downloaded and compiled.

For Windows user, you need manual to install `Protobufs` first, pleas follow the following guide:

*   Head to the [protoc releases page](https://github.com/google/protobuf/releases)

*   Download the latest `protoc-*-*.zip` release (e.g. `protoc-3.12.3-win64.zip` for 64-bit Windows)

*   Extract the contents of the downloaded `protoc-*-*.zip` in a directory `<PATH_TO_PB>` of your choice (e.g. `C:\Program Files\Google Protobuf`)

*   Add `<PATH_TO_PB>\bin` to your `Path` environment variable (see [Environment Setup](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#set-env))

#### Install WSL for running bash script
[Install WSL | Microsoft Docs](https://docs.microsoft.com/en-us/windows/wsl/install)
  
### 1. init 

```bash
./init.sh
cd workspace # after initiation the follow steps are all run in the workspace folder
```

### 2. prepare dataset

Create a folder for your dataset and put the tfrecord dataset files and label map file into the data folder. The file name should be as follow:
```
├── data
│   ├── custom01
│   │   ├── label_map.pbtxt
│   │   ├── test.tfrecord
│   │   ├── train.tfrecord
│   │   └── valid.tfrecord
│   └── oi01
│       ├── label_map.pbtxt
│       ├── train.tfrecord
│       └── valid.tfrecord
```

If you train on colab, you may want to sync the dataset from Google Drive. Check [Sync data with Google Drive](#sync-data-with-google-drive)

Create a `.env` in the `workspace` folder and set `DATASET_DIR` to the dataset you want. For example:
```ini
DATASET_DIR=data/custom01
```

### 3. download a pre-train model
Go to [tf2_detection_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), copy the model file link you want to down then run ./download_model.sh {model_link}. for example:

```bash
./download_model.sh http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
# pretrain model in pre_trained_models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
# copy and update the pipeline.config to models/ssd_mobilenet_v2_fpnlite_320x320/pipeline.config
```

This will download and extract the pre-train model into the `pre_trained_models` folder and copy the `pipeline.config` file into the coresponding `models` folder.

The script will update the `num_classes`, `fine_tune_checkpoint` and some of other configurations related to dataset. You can manually update more configurations.  Check this article for possible configurationss for model improvement
[TensorFlow Object Detection API: Best Practices to Training, Evaluation & Deployment - neptune.ai](https://neptune.ai/blog/tensorflow-object-detection-api-best-practices-to-training-evaluation-deployment)


### 4. select a model for training
   
```bash
./select_model.sh
```

This will create or update the `.env` file in the `workspace` folder.

### 5. train the selected model

```bash
./train_model.sh [--restart]
```
If you want the previously trained checkpoints, you can add `--restart` argument, it will delete all the checkpoints before training.

### 6. evaluate the model on the validation set
   
```bash
./evaluate_model.sh
```
You can run this while training.

### 7. export a trained model
```bash
./export_model.sh
```
This will export the trained model into `exported_models` folder.

### 8. test models on test dataset

* open /config/default.yml and change the `pipeline_config_path` and `checkpoint_dir` to point to your model
* run /object_detect.py 

```
Usage:
python object_detect.py [options]

General options:
-f, --config <path>         path to config file
-i, --inputpath <path>      path to test images folder
-o, --outputpath <path>     path to output folder
-e, --export_images         export images with bounding box to output folder
-c, --camera                use camera instead of images from directory
```

## Sync data with Google Drive
You can use the `sync_workspace_with_drive.sh` in the `workspace` folder to synchronize data from or to Google Drive.

Note: In colab, you need to mount the google drive into the colab runtime machine. In local, you need to use Google Back and Sync Application to mount the google drive into local drive.

Example usage:
Suppose there is a `supermarket_detection_workspace` in Google Drive. After mounting Google Drive into the colab machine, the corresponding folder is `/content/drive/MyDrive/supermarket_detection_workspace`.  In the workspace folder, run the following command to sync data.

```bash
# ./sync_workspace_with_drive.sh [folder] [--form] [--to] [--no-delete]
./sync_workspace_with_drive.sh --from # sync the whole workspace folder from google drive
./sync_workspace_with_drive.sh data --from # sync the only the data folder from google drive
./sync_workspace_with_drive.sh data --from --no-delete 
# by default the script will detele not exist files in target folder, add --no-delete if you don't want to delete

# after training a model in colab, sync the train result to google drive
./sync_workspace_with_drive.sh models/ssd_mobilenet_v2_fpnlite_320x320 --to 
```

In local, suppose `/Users/username/SMU/supermarket_detection_workspace` is the folder sync by `Backup and Sync`.
Add the follow line into `.env`.
```ini
DRIVER_DIR_PATH=/Users/username/SMU/supermarket_detection_workspace
```

Copy train result from google drive
```bash
./sync_workspace_with_drive.sh models/ssd_mobilenet_v2_fpnlite_320x320 --from
```

## Training on TPU

You booster the training by using TPU in Colab. To do this you need to put the dataset and pre-trained model in Google Cloud Storage and update the relative file path in the `pipeline.config`. 

You should set `GS_PATH_PREFIX` and `DATASET_DIR` in the workspace `.env` file. For example:

```ini
GS_PATH_PREFIX=gs://kk17_ml_data/supermarket_detection
DATASET_DIR=gs://kk17_ml_data/supermarket_detection/data/custom01
```

After that you can run the training and evaluation script with `--tpu` option. For example:

```bash
./train_model.sh --tpu
./evaluation_model.sh --tpu
./tensorboard_model.sh --tpu
```

## Build a dataset for object detection

Use fiftyone to get the require class from open image v6 dataset and build a TFOjbectDetectionDataset. Check notebooks: [building_dataset.ipynb](https://github.com/kk17/supermark_det/blob/main/notebooks/building_dataset.ipynb)

## Train a face detection model
The base for the scripts and models can be obtained in [R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X](https://github.com/R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X), but the codes have been adapted for this project.

Get the adapted pre-trained weights in [Google Drive](https://drive.google.com/drive/folders/1wFknI4ydODah3YzpSw_MFjKw1Osk3jXS?usp=sharing), and paste in `/facenet` folder

### 1. prepare face images

Download the face images and put in subfolders in the `/facenet/Faces` directory, with each person's name as the subfolder name

### 2. run training code
```
python ./facenet/train_v2.py
```
This will update `/facenet/encodings/encodings.pkl` and `/facenet/facenet_keras_weights.h5` according to the new face dataset.

### 3. test model 
run ./face_detect.py to test the model. Omit the inputpath argument to use camera instead.

```
Usage:
python object_detect.py [options]

General options:
-i, --inputpath <path>      path to test video input file, will use camera if not set
-o, --outputpath <path>     path to test video output file, will not export video if not set
-d, --detection_interval    detect face every specified number of second(s) (default=0.5s)
-s, --show                  display the labeled video as it is being processed
-m, --multi_thread          use multithread processing to speed up
```

## Reference
### Object detection

- [Training Custom Object Detector — TensorFlow 2 Object Detection API tutorial documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
- [Training and serving a realtime mobile object detector in 30 minutes with Cloud TPUs — The TensorFlow Blog](https://blog.tensorflow.org/2018/07/training-and-serving-realtime-mobile-object-detector-cloud-tpus.html)
### Face Recognition

- [R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X](https://github.com/R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X)