# unzip_val.py
from scipy import io
import os
import shutil

def move_valimg(val_dir='./val', devkit_dir='./ILSVRC2012_devkit_t12'):
    # 加载 synset 映射
    synset = io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))
    # 读取验证集标签
    with open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt')) as f:
        labels = [int(line.strip()) for line in f]

    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        val_id = int(filename.split('_')[-1].split('.')[0])
        ILSVRC_ID = labels[val_id-1]
        WIND = synset['synsets'][ILSVRC_ID-1][0][1][0]

        output_dir = os.path.join(root, WIND)
        os.makedirs(output_dir, exist_ok=True)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))

if __name__ == '__main__':
    move_valimg()
