import numpy as np
import torch
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader

def Collate_batch(batch):
    text_list, label_list, genre_list, age_list = [], [], [], []
    for (_, _text, _, _label, _age, _genre) in batch:
        label_list.append(_label)
        genre_list.append(_genre)
        age_list.append(_age)
        if len(_text)>1500:
            text_list.append(torch.tensor(_text[:1500], dtype=torch.long))
        else:
            text_list.append(torch.tensor(np.concatenate([_text, np.zeros(1500-len(_text), dtype=np.long)]) , dtype=torch.long))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    genre_list = torch.tensor(genre_list, dtype=torch.int64)
    age_list = torch.tensor(age_list, dtype = torch.int64)
    text_list = torch.stack(text_list)
    return text_list, label_list, age_list, genre_list

if __name__ == '__main__':
    
    train_transform = create_train_transform(True,True,False,True, size=params['size'], bright=False)
    train_dataset = VideoDataset(params['dataset'],size=params['size'], mode='train', play_time=params['clip_len'],frame_sample_rate=params['frame_sample_rate'], transform=train_transform, sub_classnum = params['num_classes'], label_num = params['label_num'], stride_num = params['stride'], use_plot = params['use_plot'], use_audio=True)
    plots = train_dataset.plots
    counter = Counter()
    tokenizer = get_tokenizer('basic_english')
    for plot in plots:
        counter.update(tokenizer(plot))
    vocab = Vocab(counter, min_freq=1)
    train_dataset.generate_text_pipeline(vocab, tokenizer)
    train_dataloader = DataLoader(
                                            train_dataset,
                                                                    batch_size=params['batch_size'],
                                                                                            shuffle=True,
                                                                                                                    num_workers=params['num_workers'], collate_fn = Collate_batch)
    
    for text, label, age, genre in train_dataloader:
        print(text.shape)
        print(label.shape)
        print(age.shape)
        print(genre.shape)

