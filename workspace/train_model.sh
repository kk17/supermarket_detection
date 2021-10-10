python model_main_tf2.py \
  --pipeline_config_path=models/efficientdet_d0_coco17_tpu/v1/pipeline.config\
  --model_dir=models/efficientdet_d0_coco17_tpu/v1\
  --checkpoint_every_n=100 \
  --num_workers=3 \
  --alsologtostderr