import os
import cv2
import glob
import shutil
import argparse

def main(args):
    idx = 0
    icdar_txt_path = args.icdar_dir + "/*.txt"
    filename_list = [os.path.basename(f)[:-4] for f in glob.glob(icdar_txt_path)]
    filename_lens = len(filename_list)
    for i, filename in enumerate(filename_list):
        print(i + 1, filename_lens, filename)
        
        img_path = os.path.join(args.icdar_dir, filename + '.jpg')
        csv_path = os.path.join(args.icdar_dir, filename + '.txt')

        if os.path.isfile(img_path):
            shutil.copyfile(img_path, os.path.join(args.image_dir, filename + '.jpg'))

            img = cv2.imread(img_path)
            img_h, img_w = img.shape[0], img.shape[1]

            label_list = []
            with open(csv_path, 'r') as file:
                for row in file:
                    info_list = row.split(',')
                    label = info_list[-1].replace('\n', '')
                    
                    left_top = list(map(int, info_list[0:2])) 
                    right_top = list(map(int, info_list[2:4])) 
                    right_bottom = list(map(int, info_list[4:6])) 
                    left_bottom = list(map(int, info_list[6:8])) 
                    
                    x = left_top[0]
                    y = left_top[1]
                    h = max(left_bottom[1] - left_top[1], right_bottom[1] - right_top[1])
                    w = max(left_top[0] - right_top[0], right_bottom[0] - left_bottom[0])

                    x_c = (x + (x+w)) / 2 / img_w
                    y_c = (y + (y+h)) / 2 / img_h
                    w_yolo = ((x+w) - x) / img_w
                    h_yolo = ((y+h) - y) / img_h

                    label_list.append([str(0), str(x_c), str(y_c), str(w_yolo), str(h_yolo)])

            path = os.path.join(args.label_dir, filename + '.txt')
            with open(path, 'w') as f:
                for label in label_list:
                    f.write(' '.join(label) + '\n')

if __name__ == "__main__":
    """
    Usage
        將ICDAR2019 train data轉換成YOLO格式
        python icdar2019_to_yolo.py --icdar_dir "../SROIE2019/0325updated.task1train(626p)" --image_dir datasets/ocr/images/train2017 --label_dir datasets/ocr/labels/train2017
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--icdar_dir', default='data_test', type=str)
    parser.add_argument('--image_dir', default='datasets/ocr/images/train2017', type=str)
    parser.add_argument('--label_dir', default='datasets/ocr/labels/train2017', type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        os.makedirs(args.image_dir)
    
    if not os.path.isdir(args.label_dir):
        os.makedirs(args.label_dir)
    
    main(args)
