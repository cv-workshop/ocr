import os
import re
import cv2
import glob
import argparse

def create_crop_data(args):
    image_height = 32
    image_weight = 800

    idx = 0
    txt_list = [f for f in glob.glob(args.txt_dir + "*.txt")]
    txt_lens = len(txt_list)
    for i, txt_path in enumerate(txt_list):
        filename = os.path.basename(txt_path)[:-4]
        img_path = args.jpg_dir + filename + '.jpg'
        print(i + 1, txt_lens, filename)

        if os.path.isfile(img_path):
            img = cv2.imread(img_path)

            with open(txt_path, 'r') as file:
                for row in file:
                    indices_object = re.finditer(pattern=',', string=row)
                    indices = [index.start() for index in indices_object]

                    postition = row[:indices[7]].split(',')
                    label = row[indices[7] + 1:]
                    label = label.replace('\n', '')
                    
                    left_top = list(map(int, postition[0:2])) 
                    right_top = list(map(int, postition[2:4])) 
                    right_bottom = list(map(int, postition[4:6])) 
                    left_bottom = list(map(int, postition[6:8]))
                    
                    x = left_top[0]
                    y = left_top[1]
                    h = max(left_bottom[1] - left_top[1], right_bottom[1] - right_top[1])
                    w = max(left_top[0] - right_top[0], right_bottom[0] - left_bottom[0])
                    
                    crop_img = img[y:y+h, x:x+w]
                    r = image_height / crop_img.shape[0]
                    dst = cv2.resize(crop_img, None, fx=r, fy=r, interpolation=cv2.INTER_CUBIC)
                    if dst.shape[1] < image_weight:
                        crop_img = cv2.copyMakeBorder(dst, 0, 0, 0, image_weight - dst.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    else:
                        crop_img = cv2.resize(dst, (image_weight, image_height), interpolation=cv2.INTER_CUBIC)

                    save_name = str(idx).zfill(6)
                    cv2.imwrite(os.path.join(args.save_dir, save_name + '.jpg'), crop_img)
                    with open(os.path.join(args.save_dir, save_name + '.txt'), 'w') as f:
                        f.write(label.upper())

                    idx += 1

if __name__ == "__main__":
    """
    Usage
        將ICDAR2019原始train data進行前處理:
        python create_data.py --save_dir data_train --jpg_dir "../SROIE2019/0325updated.task1train(626p)/" --txt_dir "../SROIE2019/0325updated.task1train(626p)/"
        將ICDAR2019原始test data進行前處理:
        python create_data.py --save_dir data_test --jpg_dir "../SROIE2019/task1_2_test(361p)/" --txt_dir "../SROIE2019/text_task1_2_test(361p)/" 
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='data_train', type=str)
    parser.add_argument('--jpg_dir', default='../SROIE2019/0325updated.task1train(626p)/', type=str)
    parser.add_argument('--txt_dir', default='../SROIE2019/0325updated.task1train(626p)/', type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    create_crop_data(args)

