import torch
import argparse
from itertools import groupby
from torch.utils.data import DataLoader
from model import CRNN
from utils import ICDARDataset, icdar_collate_fn

def evaluate(crnn, dataloader, blank_label):
    tot_count = 0
    tot_correct = 0
    for idx, data in enumerate(dataloader):
        print('Test Batch Idx: {}'.format(idx))
        images, targets, target_lengths = [d.cuda() for d in data]

        logits = crnn(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        tot_count += batch_size

        _, max_index = torch.max(log_probs, dim=2)
        targets = targets.cpu().numpy().tolist()
        target_lengths = target_lengths.cpu().numpy().tolist()

        target_length_counter = 0
        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
            prediction = prediction.numpy().tolist()
            
            target = targets[target_length_counter:target_length_counter + target_lengths[i]]
            target_length_counter += target_lengths[i]

            if len(prediction) == len(target) and prediction == target:
                tot_correct += 1

    evaluation = {
        'correct': tot_correct,
        'count': tot_count,
        'acc': round(tot_correct / tot_count, 4)
    }
    return evaluation

###################################################################################
def main(args):
    CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`{|}~·"
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    cpu_workers = 8
    test_batch_size = 16
    num_classes = len(CHARS) + 1
    blank_label = 0
    image_height = 32
    image_weight = 800
    cnn_output_width = 128
    rnn_hidden = cnn_output_width * 4
    use_gpu = torch.cuda.is_available()

    ### 產生ICDARataset、torch dataloader
    test_dataset = ICDARDataset(img_dir=args.test_dir, txt_dir=args.test_dir, 
                                img_height=image_height, img_width=image_weight)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, 
                             shuffle=False, num_workers=cpu_workers, collate_fn=icdar_collate_fn)
    
    ### 建立CRNN model
    model = CRNN(1, image_height, image_weight, num_classes, 
            map_to_seq_hidden=cnn_output_width, 
            rnn_hidden=rnn_hidden, 
            leaky_relu=True)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    evaluation = evaluate(model, test_loader, blank_label)
    print('Test Evaluation: correct={correct}, count={count}, acc={acc}'.format(**evaluation))

if __name__ == "__main__":
    """
    Usage
        使用訓練好的CRNN對data_test做預測, 計算accuracy
        python predict.py --test_dir data_test --model_path ./checkpoints/crnn_epoch100_testloss0.021956.pt
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='data_test', type=str)
    parser.add_argument('--model_path', default='./checkpoints/crnn_epoch100_testloss0.021956.pt', type=str)
    args = parser.parse_args()
    
    main(args)
