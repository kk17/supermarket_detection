
deepcocov2 - v6 Tfrecord
==============================

This dataset was exported via roboflow.ai on October 17, 2021 at 6:09 AM GMT

It includes 42 images.
Apples-Banana-Orange-Cerealbox-Book are annotated in Tensorflow TFRecord (raccoon) format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
Outputs per training example: 3
90° Rotate: Clockwise, Counter-Clockwise
Hue: Between -25° and +25°
Bounding Box: Rotation: Between -15° and +15°
