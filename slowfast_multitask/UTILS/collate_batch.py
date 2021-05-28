import torch
def Collate_batch(batch):
    video_list, text_list, offsets, label_list, genre_list, age_list = [],[],[0],[], [], []
    for (_video, _text, _label, _age, _genre) in batch:
        label_list.append(_label)
        genre_list.append(_genre)
        age_list.append(_age)
        video_list.append(torch.tensor(_video,dtype=torch.float))
        text_list.append(torch.tensor(_text, dtype=torch.long))
        offsets.append(_text.shape[0])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    genre_list = torch.tensor(genre_list, dtype=torch.int64)
    age_list = torch.tensor(age_list, dtype = torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    video_list = torch.stack(video_list)
    return video_list, text_list, offsets, label_list, age_list, genre_list
'''
if __name__ == '__main__':
    
    train_transform = create_train_transform(True,True,False,True, size=params['size'], bright=False)
    train_dataset = VideoDataset(params['dataset'],size=params['size'], mode='train', play_time=params['clip_len'],frame_sample_rate=params['frame_sample_rate'], transform=train_transform, sub_classnum = params['num_classes'], label_num = params['label_num'], stride_num = params['stride'], use_plot = params['use_plot'])
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
                                                                                                                    num_workers=params['num_workers'], collate_fn = collate_batch)

    for video, text, offset, label in train_dataloader:
        print(video.shape)
        print(text.shape)
        print(offset.shape)
        print(label.shape)
'''
