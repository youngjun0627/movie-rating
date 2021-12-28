import torch

if __name__=='__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    x = torch.rand(1,3,1024,112,112)
    for a in x[:,:,0:1024:16,:,:]:
        print(a.shape)
