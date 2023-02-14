import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import numpy as np
from warnings import warn
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from .utils import add_colorbar


class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                 "It may cause confusion when interpreting the results. " \
                 "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader1))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):

            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                X = feat1.flatten(1)
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    Y = feat2.flatten(1)
                    L = Y @ Y.t()
                    L.fill_diagonal_(0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        levels = np.linspace(0.0, 1.0, 20)
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma', vmin=0.0, vmax=1.0, levels=levels)
        im.set_clim(0.0, 1.0)
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        im.set_clim(0.0, 1.0)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()


class EnsembleCKA(object):
    def __init__(self,
                 models: List[nn.Module],
                 model_names: Optional[List[str]] = None,
                 model_layers: Optional[List[List[str]]] = None,
                 device: str ='cpu'):
        """

        @TODO update docs
        :param model1: (List[nn.Module]) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.models = models
        self.device = device

        self.models_info = []

        # emit warn
        for model in self.models:
            if len(list(model.modules())) > 150 and model_layers is None:
                warn("Model " + model.name + " seems to have a lot of layers. " \
                "Consider giving a list of layers whose features you are concerned with " \
                "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        if model_names is None:
            for model in self.models:
                self.models_info.append({
                    'Name': model.__repr__().split('(')[0],
                    'Layers': []
                })
        else:
            for name in model_names:
                self.models_info.append({
                    'Name': name,
                    'Layers': []
                })

        self.model_features = {}
        self.model_layers = model_layers

        self._insert_hooks()
        for ind, model in enumerate(self.models):
            self.models[ind] = model.to(self.device)
            self.models[ind].eval()

    def _log_layer(self,
                   ind: int,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if ind not in self.model_features:
            self.model_features[ind] = {}
        self.model_features[ind][name] = out

    def _insert_hooks(self):
        for ind, model in enumerate(self.models):
            for name, layer in model.named_modules():
                if self.model_layers is not None:
                    # print(name)
                    if name in self.model_layers[ind]:
                        self.models_info[ind]['Layers'] += [name]
                        layer.register_forward_hook(partial(self._log_layer, ind, name))
                else:
                    self.models_info[ind]['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, ind, name))

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()


    def compare(self,
                dataloader1: DataLoader) -> None:
        """
        Computes the feature similarity between the models on the
        given dataset.
        :param dataloader1: (DataLoader)
        """

        for info in self.models_info:
            info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]

        # ensure module sizes are equal
        D = []
        if self.model_layers is not None:
            D = [len(layers) for layers in self.model_layers]
        else:
            for model in self.models:
                D.append(len(list(model.modules())))

        if any([D[0] != f for f in D]):
            raise RuntimeError('All ensemble models must have the same number of modules')

        self.hsic_matrix = torch.zeros(D[0], D[0])

        num_batches = len(dataloader1)
        num_compare = (len(self.models)**2) - len(self.models)
        
        # @TODO look at CKA eq and find ways to reduce computation/parallize
        for m1, model1 in tqdm(enumerate(self.models), desc='| Models |', position=0, total=len(self.models)):
            for m2, model2 in enumerate(self.models):
                if m1 == m2:
                    continue

                hsic_matrix = torch.zeros(D[0], D[0], 3)
                for (x1, *_) in tqdm(dataloader1, desc="| Comparing features |", total=num_batches, position=1):
                    # capture all features
                    self.model_features = {}
                    x1 = x1.to(self.device)
                    _ = model1(x1)
                    _ = model2(x1)

                    # now create average HSIC across all models
                    for i, (name1, feat1) in enumerate(self.model_features[m1].items()):
                        X = feat1.flatten(1)
                        K = X @ X.t()
                        K.fill_diagonal_(0.0)
                        hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                        for j, (name2, feat2) in enumerate(self.model_features[m2].items()):
                            Y = feat2.flatten(1)
                            L = Y @ Y.t()
                            L.fill_diagonal_(0)
                            assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                            hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                            hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

                hsic_matrix = hsic_matrix[:, :, 1] / (hsic_matrix[:, :, 0].sqrt() * hsic_matrix[:, :, 2].sqrt())
                assert not torch.isnan(hsic_matrix).any(), "HSIC computation resulted in NANs"

                # add to average
                # print(self.hsic_matrix.shape, hsic_matrix.shape)
                self.hsic_matrix += hsic_matrix / float(num_compare)
            
    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')

        # Loop over data dimensions and create text annotations.
        for i in range(self.hsic_matrix.shape[0]):
            for j in range(self.hsic_matrix.shape[1]):
                try:
                    value = self.hsic_matrix[i, j].item()
                except:
                    value = float(self.hsic_matrix[i, j])
                text = ax.text(j, i, '%.2f' % float(value),
                            ha="center", va="center", color="w")

        ax.set_xlabel(f"Layers Ensemble", fontsize=15)
        ax.set_ylabel(f"Layers Ensemble", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"Average CKA of Ensemble", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()