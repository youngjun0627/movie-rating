import torch

if __name__=='__main__':
    x = torch.rand(1,3,1024,112,112)
    for a in x[:,:,0:1024:16,:,:]:
        print(a.shape)
