import os
import glob
import _init_paths


def gen_data_path_mft21sample(root_path):
    mot_path = 'OptMFT_Samples/images/train'
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) ] #if s.endswith('r1')
    with open('/data/liweiran/CMFTNet/src/data/samples.train', 'w') as f:
        for seq_name in seq_names:
            seq_path = os.path.join(real_path, seq_name)
            seq_path = os.path.join(seq_path, 'img1')
            images = sorted(glob.glob(seq_path + '/*.PNG'))
            len_all = len(images)
            for i in range(len_all):
                image = images[i]
                print(image[32:], file=f)
    f.close()



if __name__ == '__main__':
    root = '/data/liweiran/CMFTNet/datasets'
    gen_data_path_mft21sample(root)