import os
import glob
import subprocess

GT_IMG_PTH = '/media/storage2/datasets/DATASET FOR RAIN REMOVAL/training (training AK-Net)/ground_truth'
RAINY_IMG_PTH = '/media/storage2/datasets/DATASET FOR RAIN REMOVAL/training (training AK-Net)/rainy_image'
OUT_DIR = '/home/mahmood/methods/Single-Image-De-Raining-Keras/dataset/rain/training_custom'


gt_img_dir = glob.glob(GT_IMG_PTH + '/*.jpg')

for i, filepath in enumerate(gt_img_dir):
    filename = os.path.basename(filepath)
    subprocess.call(['ffmpeg', '-i', os.path.join(GT_IMG_PTH, filename), '-i',  os.path.join(RAINY_IMG_PTH, filename), '-filter_complex', 'hstack', os.path.join(OUT_DIR, str(i) + '.jpg')])