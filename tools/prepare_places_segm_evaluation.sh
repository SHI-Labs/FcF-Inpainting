# 0. folder preparation
rm -r -f datasets/places2_dataset/evaluation/segm_hires/
mkdir -p datasets/places2_dataset/evaluation/segm_hires/
mkdir -p datasets/places2_dataset/evaluation/random_segm_256/

# 1. sample 10000 new images
OUT=$(python3 tools/eval_segm_sampler.py)
echo ${OUT}

echo "Preparing images..."
SEGM_FILELIST=$(cat datasets/places2_dataset/eval_random_segm_files.txt)
for i in $SEGM_FILELIST
do
    $(cp ${i} datasets/places2_dataset/evaluation/segm_hires/)
done



# 2. generate segmentation masks

python3 tools/gen_random_segm_masks.py \
    training/data/configs/segm_256.yaml \
    datasets/places2_dataset/evaluation/segm_hires/ \
    datasets/places2_dataset/evaluation/random_segm_256
