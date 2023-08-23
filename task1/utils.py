import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ICDARDataset(Dataset):
    CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`{|}~·"
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, img_dir=None, txt_dir=None, img_height=32, img_width=100):
        self.paths = []
        self.texts = []
        img_dir_glob = os.path.join(img_dir + "/*.jpg")
        filenames = [os.path.basename(f)[:-4] for f in glob.glob(img_dir_glob)]
        for idx, filename in enumerate(filenames):
            img_path = os.path.join(img_dir, filename + '.jpg')
            txt_path = os.path.join(txt_dir, filename + '.txt')

            f = open(txt_path)
            text = f.read()
            text = text.replace(" ", "")
            if len(text) != 0:
                self.paths.append(img_path)
                self.texts.append(txt_path)

        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        txt_path = self.texts[index]
        f = open(txt_path)
        text = f.read()
        target = [self.CHAR2LABEL[c] for c in text if c in self.CHAR2LABEL]
        target_length = [len(target)]
        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)

        return image, target, target_length


def icdar_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
