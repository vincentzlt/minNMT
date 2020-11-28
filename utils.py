import torch


def average_state_dicts(state_dicts):
    keys = set(state_dicts[0].keys())
    assert all(set(d.keys()) == keys for d in state_dicts)

    new_state_dict = dict()
    for k in keys:
        value = torch.stack([d[k] for d in state_dicts]).mean(0)
        new_state_dict[k] = value
    return new_state_dict