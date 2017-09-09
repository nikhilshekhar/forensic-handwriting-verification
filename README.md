#GOAL:
Given two samples of handwriting, the model learn't can be used to predict if they belong to the same person or not

# TODO
# Interesting links
- Try SELU - https://github.com/bioinf-jku/SNNs/blob/master/selu.py
- Image matching based on features - http://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W03/papers/Lin_Deep_Learning_of_2015_CVPR_paper.pdf
- Try self-normalizing networks
- Image augmentation 
- Proper corpping of images

# Install steps for python 3.6 with conda and compatible tensorflow
conda info --envs
conda create --name tfpy python=3.6
source activate tfpy
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.2.0-py3-none-any.whl
