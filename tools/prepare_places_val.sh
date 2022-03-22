# 0. folder preparation
rm -r -f datasets/places2_dataset/evaluation/val_hires/
mkdir -p datasets/places2_dataset/evaluation/val_hires/
mkdir -p datasets/places2_dataset/evaluation/random_val/

# 1. sample 10000 new images
OUT=$(python3 tools/val_sampler.py)
echo ${OUT}

echo "Preparing images..."
FILELIST=$(cat datasets/places2_dataset/val_random_files.txt)
for i in $FILELIST
do
    $(cp ${i} datasets/places2_dataset/evaluation/val_hires/)
done



# 2. generate all kinds of masks

python3 tools/gen_masks.py \
    --img_data=datasets/places2_dataset/evaluation/val_hires/ \
    --msk_ratio=0.0 \
    --msk_ratio=0.7 \
    --msk_type=datasets/places2_dataset/evaluation/random_val
