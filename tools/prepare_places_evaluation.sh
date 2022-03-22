# 0. folder preparation
rm -r -f datasets/places2_dataset/evaluation/hires/
mkdir -p datasets/places2_dataset/evaluation/hires/
mkdir -p datasets/places2_dataset/evaluation/random_thick_256/
mkdir -p datasets/places2_dataset/evaluation/random_thin_256/
mkdir -p datasets/places2_dataset/evaluation/random_medium_256/
mkdir -p datasets/places2_dataset/evaluation/free_form_256/

# 1. sample 30000 new images
OUT=$(python3 tools/eval_sampler.py)
echo ${OUT}

echo "Preparing images..."
FILELIST=$(cat datasets/places2_dataset/eval_random_files.txt)
for i in $FILELIST
do
    $(cp ${i} datasets/places2_dataset/evaluation/hires/)
done



# 2. generate all kinds of masks

python3 tools/gen_masks.py \
    --img_data=datasets/places2_dataset/evaluation/hires/ \
    --lama_cfg=training/data/configs/thin_256.yaml \
    --msk_type=datasets/places2_dataset/evaluation/random_thin_256

python3 tools/gen_masks.py \
    --img_data=datasets/places2_dataset/evaluation/hires/ \
    --lama_cfg=training/data/configs/thick_256.yaml \
    --msk_type=datasets/places2_dataset/evaluation/random_thick_256

python3 tools/gen_masks.py \
    --img_data=datasets/places2_dataset/evaluation/hires/ \
    --lama_cfg=training/data/configs/medium_256.yaml \
    --msk_type=datasets/places2_dataset/evaluation/random_medium_256

python3 tools/gen_masks.py \
    --img_data=datasets/places2_dataset/evaluation/hires/ \
    --msk_ratio=0.0 \
    --msk_ratio=0.7 \
    --msk_type=datasets/places2_dataset/evaluation/free_form_256
