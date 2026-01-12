import os
import torch
from model import TimeBridge


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimeBridge': TimeBridge,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        import torch
        if self.args.use_gpu:
            if getattr(self.args, "use_mps", False):
                device = torch.device("mps")
                print("Use MPS")
            else:
                device = torch.device(f"cuda:{self.args.gpu}")
                print("Use CUDA")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device


    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
