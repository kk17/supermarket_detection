from supermarket_detection import dataset_utils

coco_classes = ["apple", "orange", "banana", "book"]
oi_classes = [c[0].upper() + c[1:] for c in coco_classes]

open_image_test_ds = dataset_utils.build_dataset(oi_classes,
                                                 split='test',
                                                 max_samples_per_class=1000,
                                                 delete_exist_dataset=False)
open_image_test_filterd_view = dataset_utils.build_dateset_view(
    open_image_test_ds,
    valid_labels=oi_classes,
    IsGroupOf=False,
    bbox_area_lower_bound=0.1)
counts = open_image_test_filterd_view.count_values(
    "detections.detections.label")
print(counts)
open_image_test_filterd_view.save()
print("dataset saved")