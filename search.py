import torch
import model


class Search:
    def __init__(self, model=model.MyTransformer):
        self.model = model

    def greedy(self, x, max_len=None):
        if max_len is None:
            max_len = int(x.size(0) * 1.1)
        state = self.model.init_state(x)
        y_step = x[:0]
        finished = torch.zeros_like(x[:1].view(-1)).bool()
        for i in range(max_len):
            y_step_prob, state = self.model.forward_step(y_step, state)
            y_step = y_step_prob.max(-1)[1]
            finished += y_step.view(-1).eq(3)
            if finished.all():
                break
        return state[1]
