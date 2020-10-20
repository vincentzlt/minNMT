from os import stat
import torch
import argparse as ap

from torch import scatter_add


def average_state_dicts(state_dicts):
    keys = set(state_dicts[0].keys())
    assert all(set(d.keys()) == keys for d in state_dicts)

    new_state_dict = dict()
    for k in keys:
        value = torch.stack([d[k] for d in state_dicts]).mean(0)
        new_state_dict[k] = value
    return new_state_dict


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('--inputs', type=str, nargs='+')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    ckpts = [torch.load(f, map_location='cpu') for f in args.inputs]
    state_dicts = [c['state_dict'] for c in ckpts]
    new_state_dict = average_state_dicts(state_dicts)
    ckpt = ckpts[0]
    ckpt['state_dict'] = new_state_dict
    torch.save(ckpt, args.output)