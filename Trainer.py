import json
from collections import defaultdict
from itertools import repeat

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from tqdm import trange

from Globals import args, output_path, device, StatsMgr
from Mesher import handle_meshes
from Modules import OReXNet


class Trainer:
    def __init__(self, csl):
        self.csl = csl

        self.module = OReXNet()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.data_loader = None

        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.scheduler_gamma)

        self.total_epochs = 0
        self.train_losses = defaultdict(list)

    def _get_loss_parts(self, xyzs, labels):
        xyzs.requires_grad_(True)

        pred_iterations = self._get_iterations_predictions(xyzs)

        grad_xyzs = torch.autograd.grad(pred_iterations[-1].sum(), [xyzs], create_graph=True)[0]

        loss_parts = {}

        # bce_loss has a sigmoid layer build in
        loss_parts['BCE'] = sum(map(self.bce_loss, pred_iterations, repeat(labels))) / len(pred_iterations)

        if args.eikonal_hinge_lambda > 0:
            norms = grad_xyzs.norm(dim=-1)
            norm_loss = torch.maximum(norms - args.hinge_alpha, torch.zeros_like(norms)).mean()
            loss_parts['eikonal_hinge'] = norm_loss * args.eikonal_hinge_lambda

        return loss_parts

    def _train_epoch(self):
        running_loss = defaultdict(int)

        for xyzs, labels in self.data_loader:

            xyzs, labels = xyzs.to(device), labels.to(device)

            loss_parts = self._get_loss_parts(xyzs, labels)

            self.optimizer.zero_grad()
            loss = sum(loss_parts.values())

            loss.backward()
            self.optimizer.step()

            # update running loss
            d = {}
            for k in loss_parts.keys():
                d[k] = running_loss[k] + loss_parts[k].item()
            running_loss = d

        if self.total_epochs > 0 and self.total_epochs % args.scheduler_step == 0:
            self.lr_scheduler.step()

        # log batch loss
        for k, v in running_loss.items():
            self.train_losses[k].append(v / len(self.data_loader.dataset))

    def _get_iterations_predictions(self, xyzs):
        return self.module(xyzs, args.n_iterations)

    def _get_batch_predictions(self, xyzs):
        return self._get_iterations_predictions(xyzs)[-1]

    def _update_used_dataset(self, data_sets, refinement_level):

        new_dataset = ConcatDataset(data_sets[0:refinement_level] if args.n_used_datasets is None \
                                        else data_sets[-args.n_used_datasets:])

        self.data_loader = DataLoader(new_dataset, batch_size=2 ** args.batch_size_exp, sampler=None,
                                      shuffle=True, pin_memory=True, num_workers=4)

    def _train_refinement_level(self, epochs, refinement_level):
        self.module.train()

        for _ in trange(epochs,
                            desc=f'Refinement level {refinement_level}/{len(args.epochs_batches) - 1} dataset={len(self.data_loader.dataset)}'):
            self._train_epoch()
            self.total_epochs += 1

    def log_train_losses(self):
        assert 'total' not in self.train_losses

        # update train_losses with the total loss
        self.train_losses['total'] = list(map(sum, zip(*self.train_losses.values())))

        for k, v in self.train_losses.items():
            if len(v) > 0:
                plt.bar(range(len(v)), np.clip(v, 0, 2 * np.percentile(v, 95)))

                plt.savefig(output_path + f"losses_{k}.png", dpi=500)
                # plt.show()
                plt.close()
                plt.cla()
                plt.clf()

        for k, v in self.train_losses.items():
            StatsMgr.setitem(f'last_loss_{k}', v[-1])

        with open(output_path + 'losses.json', 'w') as f:
            f.write(json.dumps(self.train_losses, default=lambda o: o.__dict__, indent=4))

    @torch.no_grad()
    def get_predictions(self, xyzs):
        self.module.eval()
        data_loader = DataLoader(xyzs, batch_size=2 ** args.batch_size_exp, shuffle=False,
                                 num_workers=4, pin_memory=True)
        label_pred = np.empty(0, dtype=float)
        for xyzs_batch in data_loader:
            xyzs_batch = xyzs_batch.to(device)
            batch_labels = self._get_batch_predictions(xyzs_batch)

            label_pred = np.concatenate((label_pred, batch_labels.detach().cpu().numpy().reshape(-1)))
        return label_pred

    def grad_wrt_input(self, xyzs):
        self.module.eval()

        data_loader = DataLoader(xyzs, batch_size=2 ** args.batch_size_exp, shuffle=False,
                                 num_workers=4, pin_memory=True)
        grads = np.empty((0, 3), dtype=float)

        for xyzs_batch in data_loader:
            xyzs_batch = xyzs_batch.to(device)
            xyzs_batch.requires_grad_(True)

            self.module.zero_grad()
            pred = self._get_batch_predictions(xyzs_batch)

            grads_batch = torch.autograd.grad(pred.mean(), [xyzs_batch])[0].detach().cpu().numpy()
            grads = np.concatenate((grads, grads_batch))
        return grads

    def load_model(self, path):
        self.module.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.module.to(device)
        self.module.eval()

    def save_model(self, path):
        torch.save(self.module.state_dict(), path)

    def train_cycle(self, data_sets_promises):
        data_sets = []
        for refinement_level, epochs in enumerate(args.epochs_batches):

            data_sets.append(data_sets_promises[refinement_level].get())
            data_sets[-1].to_ply(f'{output_path}/datasets/gen{refinement_level}.ply')

            self._update_used_dataset(data_sets, refinement_level)

            with StatsMgr.timer('train', refinement_level):
                self._train_refinement_level(epochs, refinement_level)

            StatsMgr['dataset_size'][refinement_level] = len(data_sets[refinement_level])

            self.save_model(output_path + f"/models/trained_model_{refinement_level}.pt")

            try:
                handle_meshes(self, refinement_level)
            except Exception as exept:
                print(exept)
