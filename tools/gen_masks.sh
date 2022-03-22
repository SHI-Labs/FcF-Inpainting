#!/bin/sh

python tools/gen_masks.py --img_data=datasets/LaMa_test_images \
        --msk_type=test \
        --lama_cfg=training/data/configs/segm_256.yaml 