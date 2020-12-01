from pytorch_lightning import Trainer
import torch
import model
import sys
from train import MT
import data
import git
import argparse as ap
from tqdm import tqdm


class Node:
    def __init__(self, score, index, state, prev=None, next=None):
        self.score = score.clone().detach()
        self.index = index.clone().detach()
        self.state = (s for s in state)
        self.prev = prev
        if next is None:
            self.next = []
        else:
            self.next = next

    def __repr__(self):
        return f'<Node {self.index.item()} {self.score.item():.2f}>'

    @staticmethod
    def collect_index(nodes):
        index = torch.stack([n.index for n in nodes], 1)
        return index

    @staticmethod
    def collect_state(nodes):
        states = [n.state for n in nodes]
        state = list(zip(*states))
        state = [torch.stack(s, dim=1) for s in state]
        return state

    @staticmethod
    def split_state(state):
        state = [s.unbind(1) for s in state]
        states = list(zip(*state))
        return states

    @property
    def nodes_sofar(self):
        nodes = [self]
        while nodes[-1].prev is not None:
            nodes.append(nodes[-1].prev)
        return list(reversed(nodes))

    def reduced_score(self, lenpen=0):
        score = torch.stack([n.score.log()
                             for n in self.nodes_sofar[1:]]).mean(0)
        lp = (((5 + len(self))**lenpen) / (6**lenpen))
        score = score / lp
        return score

    def __len__(self):
        return len(self.nodes_sofar)


class Search:
    def __init__(self, model=model.Transformer(), lenpen=0):
        if isinstance(model, torch.nn.Module):
            self.model = model
        elif isinstance(model, list) and all(
                isinstance(m, torch.nn.Module) for m in model):
            self.model = self.average_models(model)
        else:
            raise ValueError(
                'Search object requires nn.Module or list of nn.Module.')

        self.lenpen = lenpen

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

    def topk(self, x, max_len=100, k=1):
        beams, finished = self.init_beams(x, k)
        with tqdm(range(max_len), leave=False) as t:
            for i in t:
                beams, finished = self.beam_search(beams, finished)
                t.set_postfix({'alive': len([n for b in beams for n in b])})
                if len([n for beam in beams for n in beam]) == 0:
                    break
        hyp = self.beam_finalize(beams, finished)
        return hyp

    def init_beams(self, x, k):
        state = self.model.init_state(x)

        states = Node.split_state(state)
        beams_bos = [[
            Node(
                torch.tensor([0]).type_as(x).float(),
                torch.tensor([2]).type_as(x), state)
        ] for state in states]

        y_step = x[:0]
        y_step_prob, state = self.model.forward_step(y_step, state)
        y_step_scores, y_step_indexs = y_step_prob.topk(k, -1)

        beams = []
        finished = []
        scores = y_step_scores.unbind(1)
        indexs = y_step_indexs.unbind(1)
        states = Node.split_state(state)
        for indexs, scores, state, beam_bos in zip(indexs, scores, states,
                                                   beams_bos):
            beam = []
            finished_beam = []
            for index, score in zip(indexs.unbind(-1), scores.unbind(-1)):
                node = Node(score, index, state, prev=beam_bos[0])
                if node.index != 3:
                    beam.append(node)
                else:
                    finished_beam.append(node)
            beams.append(beam)
            finished.append(finished_beam)
        return beams, finished

    def beam_search(self, beams, finished):
        nodes = [node for beam in beams for node in beam]
        state = Node.collect_state(nodes)
        y_step = Node.collect_index(nodes)
        y_step_prob, state = self.model.forward_step(y_step, state)
        y_step_scores, y_step_indexs = y_step_prob.sort(-1, descending=True)

        new_beams = []
        scores = y_step_scores.unbind(1)
        indexs = y_step_indexs.unbind(1)
        states = Node.split_state(state)
        node_idx = 0
        for beam in beams:
            new_beam = []
            for node in beam:
                scores_ = scores[node_idx]
                indexs_ = indexs[node_idx]
                state_ = states[node_idx]
                for i in range(len(beam)):
                    new_node = Node(scores_[:, i],
                                    indexs_[:, i],
                                    state_,
                                    prev=node)
                    node.next.append(new_node)
                    node.state = None
                    new_beam.append(new_node)
                node_idx += 1
            new_beams.append(new_beam)
        assert len(scores) == len(indexs) == len(states) == node_idx

        for i in range(len(beams)):
            beam = beams[i]
            new_beam = new_beams[i]
            finished_beam = finished[i]

            new_beam = sorted(new_beam,
                              key=lambda n: n.reduced_score(self.lenpen),
                              reverse=True)[:len(beam)]
            new_beams[i] = [n for n in new_beam if n.index != 3]
            finished[i] = finished_beam + [n for n in new_beam if n.index == 3]

        return new_beams, finished

    def beam_finalize(self, beams, finished):
        for i, finished_beam in enumerate(finished):
            finished[i] = finished_beam + beams[i]
            finished[i] = sorted(finished[i],
                                 key=lambda n: n.reduced_score(self.lenpen),
                                 reverse=True)
        finished = [b[0].nodes_sofar for b in finished]
        finished = [[n.index for n in nodes] for nodes in finished]

        max_len = max(len(nodes) for nodes in finished)
        batch_size = len(finished)
        index = torch.zeros(batch_size, max_len).long()
        for i, ids in enumerate(finished):
            index[i][:len(ids)] = torch.tensor(ids)
        return index.T


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        default=
        '/storage07/user_data/zhanglongtu01/minNMT/data/wmt14.en-de.stanford/newstest2014.en.tok.id'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default=
        '/storage07/user_data/zhanglongtu01/minNMT/exp/run_8/lightning_logs/version_0/checkpoints/step.109500.ckpt'
    )
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=25000)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--lenpen', type=float, default=0.6)
    args = parser.parse_args()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print('commit #: ', sha, file=sys.stderr)

    mt_model = MT.load_from_checkpoint(args.ckpt)
    mt_model.hparams.beam_size = args.beam_size
    mt_model.hparams.lenpen = args.lenpen
    mt_model.search = Search(mt_model.model, args.lenpen)
    mt_model = mt_model.cuda(args.gpu).eval()

    dataset = data.Dataset('en', 'de', args.input, args.input, args.input,
                           args.input, args.input, args.input, args.batch_size,
                           4)
    dataset.setup()
    # trainer = Trainer(gpus=args.gpus)
    # hyp_idss = trainer.test(mt_model, datamodule=dataset, verbose=False)
    # for l in hyp_idss:
    #     print(' '.join([str(i) for i in l]))

    for batch in tqdm(dataset.test_dataloader()):
        x = batch[mt_model.hparams.src_lang].cuda(args.gpu)
        y_hyp = mt_model.search.topk(
            x,
            int(x.size(0) * 2),
            args.beam_size,
        )
        idss = dataset.unpad(y_hyp)
        for ids in idss:
            print(' '.join([str(i) for i in ids]))