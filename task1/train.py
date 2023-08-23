import os
import torch
import argparse
import torch.nn as nn
from itertools import groupby
from torch.utils.data import DataLoader
from model import CRNN
from utils import ICDARDataset, icdar_collate_fn

def train_batch(crnn, data, optimizer, criterion):
    images, targets, target_lengths = [d.cuda() for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5)
    optimizer.step()
    return loss.item(), log_probs

def evaluate(crnn, dataloader, criterion, blank_label):
    tot_count = 0
    tot_loss = 0
    tot_correct = 0

    for data in dataloader:
        images, targets, target_lengths = [d.cuda() for d in data]

        logits = crnn(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        tot_count += batch_size
        tot_loss += loss.item()

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
        'loss': tot_loss / tot_count,
        'correct': tot_correct,
        'count': tot_count
    }
    return evaluation

def main(args):
    ### parameters
    CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`{|}~·"
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    cpu_workers = 8
    epochs = 100
    train_batch_size = 16
    valid_batch_size = 8
    lr = 0.0001
    show_interval = 100
    save_interval = 10
    num_classes = len(CHARS) + 1
    blank_label = 0
    image_height = 32
    image_weight = 800
    cnn_output_width = 128
    rnn_hidden = cnn_output_width * 4
    reload_checkpoint = None
    use_gpu = torch.cuda.is_available()

    ### 產生ICDARataset、torch dataloader
    train_dataset = ICDARDataset(img_dir=args.train_dir, txt_dir=args.train_dir, 
                                 img_height=image_height, img_width=image_weight)
    test_dataset = ICDARDataset(img_dir=args.test_dir, txt_dir=args.test_dir, 
                                img_height=image_height, img_width=image_weight)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, 
                            shuffle=True, num_workers=cpu_workers, collate_fn=icdar_collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=valid_batch_size, 
                            shuffle=False, num_workers=cpu_workers, collate_fn=icdar_collate_fn)
    
    ### 建立CRNN model
    crnn = CRNN(1, image_height, image_weight, num_classes, 
                map_to_seq_hidden=cnn_output_width, 
                rnn_hidden=rnn_hidden, 
                leaky_relu=True)

    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint))

    model = crnn.cuda()
    criterion = nn.CTCLoss(blank=blank_label, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    ### train model
    for epoch in range(1, epochs + 1):
        i = 1
        tot_train_loss = 0.
        tot_train_count = 0
        tot_train_correct = 0
        for train_data in train_loader:
            loss, log_probs = train_batch(model, train_data, optimizer, criterion)
            targets = train_data[1].numpy().tolist()
            target_lengths = train_data[2].numpy().tolist()
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size

            _, max_index = torch.max(log_probs, dim=2)
            target_length_counter = 0
            for j in range(train_size):
                raw_prediction = list(max_index[:, j].detach().cpu().numpy())
                prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
                prediction = prediction.numpy().tolist()
                
                target = targets[target_length_counter:target_length_counter + target_lengths[j]]
                target_length_counter += target_lengths[j]

                if len(prediction) == len(target) and prediction == target:
                    tot_train_correct += 1
            
            if i % show_interval == 0:
                    print('Epoch: {}, Batch Idx: {}'.format(epoch, i))
            i += 1
        print("Epoch: {}, Training Loss: {:.6f}, Training Correct: {} / {}".format(epoch, tot_train_loss / tot_train_count, tot_train_correct, tot_train_count))
        
        ### 用test data驗證model
        evaluation = evaluate(model, test_loader, criterion, blank_label)
        print('valid_evaluation: loss={loss}, correct={correct}, count={count}'.format(**evaluation))

        ### 根據save_interval來決定是否存模型
        if epoch % save_interval == 0:
            prefix = 'crnn'
            test_loss = evaluation['loss']
            save_model_path = os.path.join(args.save_dir, f'{prefix}_epoch{str(epoch).zfill(3)}_testloss{test_loss:.6f}.pt')
            torch.save(crnn.state_dict(), save_model_path)


if __name__ == "__main__":
    """
    Usage
        使用data_train、data_test來訓練CRNN model, 並存模型在checkpoints dir
        python train.py --train_dir data_train --test_dir data_test --save_dir checkpoints
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data_train', type=str)
    parser.add_argument('--test_dir', default='data_test', type=str)
    parser.add_argument('--save_dir', default='checkpoints', type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    main(args)