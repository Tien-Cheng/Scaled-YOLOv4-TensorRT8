import torch
import struct
from sys import argv
from utils.torch_utils import select_device


def generate_wts():
    weights = argv[1]
    # Initialize
    device = select_device('cpu')
    # Load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()

    f = open('yolov4-p6.wts', 'w')
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')

if __name__ == "__main__":
    generate_wts()