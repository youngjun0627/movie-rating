import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
'''
ref : https://github.com/slaysd/pytorch-sentiment-analysis-classification/blob/master/model.py
'''

class LSTM_with_Attention(nn.Module):
    def __init__(self, 
                vocab_size=-1, 
                embedding_dim=300, 
                hidden_dim=256, 
                class_num = 4,
                label_num = 4,
                n_layers = 2,
                use_bidirectional=True,
                use_dropout = True):
        super(LSTM_with_Attention, self).__init__()
        self.embedding_dim = embedding_dim
        #self.embedding = None
        self.embedding = nn.Embedding(1500,300,padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, 
                        batch_first = True,
                        bidirectional = use_bidirectional,
                        num_layers = n_layers,
                        dropout = 0.7 if use_dropout and n_layers>1 else 0.) 
        input_dim = hidden_dim * 2 if use_bidirectional else hidden_dim
        #self.bn = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([nn.Linear(input_dim, class_num) for _ in range(label_num)])
        # genre fc, age fc #
        self.genrefc = nn.Linear(input_dim, 9)
        self.agefc = nn.Linear(input_dim, 4)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    
    def init_text_weights(self, vocab_size=None, weights = None):
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx = 0)
        self.embedding.load_state_dict({'weight':torch.from_numpy(weights)})
        self.embedding.weight.requires_grad = False 


    def attention(self, lstm_output, final_state):
        #print(lstm_output.shape)
        #lstm_output = lstm_output.squeeze(0)
        #print(lstm_output.shape)
        #hidden = final_state.squeeze(0)
        hidden = final_state
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        return torch.bmm(lstm_output.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
    
    def spatial_dropout(self, x):
        x = x.permute(0,2,1)
        x = F.dropout2d(x, 0.4, training=self.training)
        x = x.permute(0,2,1)
        return x

    def forward(self, x):
        embedded = self.embedding(x)
        #embedded = self.spatial_dropout(embedded)
        output, _ = self.rnn(embedded)
        #output = self.bn(output)
        attn_output = self.attention(output, output.transpose(0,1)[-1])
        #logit = attn_output.squeeze(0)
        logit = self.dropout(attn_output)
        return logit
        '''
        classes = []
        for fc in self.fcs:
            classes.append(fc(logit))
        classes = torch.stack(classes)
        genre = self.genrefc(logit)
        age = self.agefc(logit)
        return classes, genre, age
        '''


if __name__ == "__main__":
    init_text_weights()
    #weight_matrix = np.load('glove_embeddings.npy', allow_pickle=True)
    #print(weight_matrix)
'''
from torchinfo import summary

if __name__=="__main__":
    train_transform = None
    da = VideoDataset(params['dataset'],size=params['size'], mode='train', play_time=params['clip_len'],frame_sample_rate=params['frame_sample_rate'], transform=train_transform, sub_classnum = params['num_classes'], label_num = params['label_num'], stride_num = params['stride'], use_plot = params['use_plot'], use_audio = params['use_audio'], use_video = params['use_video'])
    model = LSTM_with_Attention()
    if True:
        plots = da.plots
        plots = set(plots)
        counter = Counter()
        tokenizer = get_tokenizer('basic_english')
        for plot in plots:
            counter.update(tokenizer(plot))
        vocab = Vocab(counter, min_freq = 1)
        da.generate_text_pipeline(vocab, tokenizer)

        #val_dataset.generate_text_pipeline(vocab, tokenizer)
        model.init_text_weights(len(vocab))
    d = []
    for _,text,_,_,_,_ in [da[0],da[1],da[2],da[3]]:
        if len(text)>60:
            d.append(torch.tensor(text[:60], dtype=torch.long))
        else:
            d.append(torch.tensor(np.concatenate([text, np.zeros(60-len(text), dtype=np.long)]) , dtype=torch.long))

    d = torch.stack(d)
    print(d.shape)
    print('========')
    print(model(d).shape)
    #summary(model, train_dataset[0][1])
'''
