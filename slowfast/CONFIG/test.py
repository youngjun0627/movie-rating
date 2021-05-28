import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str)

args = parser.parse_args()
if args.mode=='ok':
    import torch
else:
    pass

print(torch.__version__)
