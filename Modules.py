from itertools import chain

import torch
from torch import nn

from Globals import args, device


def _initializer(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.uniform_(m.bias, 0, 1)


def _get_MLP_layers(n_neurons):
    neurons = [nn.Linear(n_neurons[i], n_neurons[i + 1]) for i in range(len(n_neurons) - 2)]
    activations = [nn.LeakyReLU() for _ in range(len(n_neurons) - 2)]
    layers = list(chain.from_iterable(zip(neurons, activations))) + [nn.Linear(n_neurons[-2], n_neurons[-1])]
    return layers


class OReXNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.xyz_embedder = get_embedder(args.embedding_freqs, learnable=True)
        self.hidden_state_embedder = get_embedder(args.hidden_state_size//2, input_dims=1, include_input=False)
        assert self.hidden_state_embedder.out_dim == args.hidden_state_size

        n_neurons = [self.xyz_embedder.out_dim + 1 + args.hidden_state_size] \
                    + [args.hidden_layers_height]*args.hidden_layers_width\
                    + [1 + args.hidden_state_size]

        self.MLP = nn.Sequential(*_get_MLP_layers(n_neurons))
        self.initial_logits = nn.Parameter(torch.empty(1))
        self.initial_hidden_states = nn.Parameter(torch.empty(args.hidden_state_size))

        self._init_weights()

        self.double()
        self.to(device)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, xyzs, n_loops):
        logits = self.initial_logits.expand(len(xyzs), 1)
        hidden_states = self.initial_hidden_states.expand(len(xyzs), len(self.initial_hidden_states))

        ret = []
        for i in range(n_loops):
            logits, hidden_states = self._forward_iteration(xyzs, logits, hidden_states,
                                                            torch.tensor([i], device=device))
            ret.append(logits)
        return ret

    def _forward_iteration(self, xyzs, logits, hidden_states, i):
        if self.hidden_state_embedder is not None:
            hidden_states = hidden_states + self.hidden_state_embedder(i)

        outputs = self.MLP(torch.cat((self.xyz_embedder(xyzs), logits, hidden_states), dim=1))
        residual = outputs[:, 0:1]
        out_hidden_states = outputs[:, 1:]

        return logits + residual, out_hidden_states

    def _init_weights(self):
        self.MLP.apply(_initializer)

        torch.nn.init.uniform_(self.initial_hidden_states, 0, 1)
        torch.nn.init.uniform_(self.initial_logits, 0, 1)


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class EmbedderWithLearnableFreqs(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        self.cos_freq = nn.Parameter(2. ** torch.linspace(0., max_freq, steps=N_freqs))
        self.sin_freq = nn.Parameter(2. ** torch.linspace(0., max_freq, steps=N_freqs))

        self.out_dim = 2 * N_freqs * self.kwargs['input_dims'] + self.kwargs['input_dims']

    def forward(self, inputs):
        cos = torch.cos(inputs[..., None] * self.cos_freq).view((len(inputs), -1))
        sin = torch.sin(inputs[..., None] * self.sin_freq).view((len(inputs), -1))
        return torch.cat((inputs, cos, sin), -1)


# adapted from https://github.com/yenchenlin/nerf-pytorch/blob/a15fd7cb363e93f933012fd1f1ad5395302f63a4/run_nerf_helpers.py#L48
def get_embedder(multires, input_dims=3, include_input=True, learnable=False):

    embed_kwargs = {
        'include_input': include_input,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'periodic_fns': [torch.sin, torch.cos],
    }

    return  EmbedderWithLearnableFreqs(**embed_kwargs) if learnable else  Embedder(**embed_kwargs)
