#!/bin/sh

python3 tools/gen_masks.py \
    --img_data=datasets/celeba-hq-dataset/visual_test_source_256/ \
    --lama_cfg=training/data/configs/thin_256.yaml \
    --msk_type=datasets/celeba-hq-dataset/visual_test_256/random_thin_256

python3 tools/gen_masks.py \
    --img_data=datasets/celeba-hq-dataset/visual_test_source_256/ \
    --lama_cfg=training/data/configs/thick_256.yaml \
    --msk_type=datasets/celeba-hq-dataset/visual_test_256/random_thick_256

python3 tools/gen_masks.py \
    --img_data=datasets/celeba-hq-dataset/visual_test_source_256/ \
    --lama_cfg=training/data/configs/medium_256.yaml \
    --msk_type=datasets/celeba-hq-dataset/visual_test_256/random_medium_256

python3 tools/gen_masks.py \
    --img_data=datasets/celeba-hq-dataset/visual_test_source_256/ \
    --msk_ratio=0.0 \
    --msk_ratio=0.7 \
    --msk_type=datasets/celeba-hq-dataset/visual_test_256/free_form_256