import os
import glob
import argparse

def create_crop_data(args):
    glob_path = args.txt_dir + "/*.txt"

    CHARS = ""
    txt_paths = [f for f in glob.glob(glob_path)]
    txts_len = len(txt_paths)
    for idx, txt_path in enumerate(txt_paths):
        print(idx + 1, txts_len, txt_path)
        with open(txt_path, 'r') as file:
            for row in file:
                row_uniq = "".join(set(row))

                CHARS += row_uniq
                CHARS = "".join(set(CHARS))
    CHARS = "".join(sorted(CHARS))
    print(CHARS)

if __name__ == "__main__":
    """
    Usage
        取得data_train資料夾內所有txt的數字、字母、符號
        python get_char_label.py --txt_dir data_train
        取得data_test資料夾內所有txt的數字、字母、符號
        python get_char_label.py --txt_dir data_train
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_dir', default='data_train', type=str)
    args = parser.parse_args()
    
    create_crop_data(args)