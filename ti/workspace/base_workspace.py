from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading
import wandb
import pdb
from datetime import datetime


def adapt_state_dict_for_ddp(state_dict):
    """Prepends 'module.' to all keys in the state dictionary."""
    return {'module.' + k: v for k, v in state_dict.items()}


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None, rank=0, world_size=1):
        self.cfg = cfg

        if output_dir is None or output_dir.lower() == 'none':
            try:
                output_dir = HydraConfig.get().runtime.output_dir
            except:
                output_dir = None
            # try:
            # except:
            #     now = datetime.now()
            #     now_str = now.strftime("%Y.%m.%d/%H.%M.%S")
            #     name = cfg.name
            #     task_name = cfg.task_name
            #     output_dir = f"data/outputs/{now_str}_{name}_{task_name}"
            #     output_dir = hydra.utils.to_absolute_path(output_dir)
        else:
            output_dir = hydra.utils.to_absolute_path(output_dir)
        self.output_dir = output_dir
        
        self._saving_thread = None
        
        self.rank = rank
        self.world_size = world_size

    # @property
    # def output_dir(self):
    #     # output_dir = self._output_dir
    #     # if output_dir is None or output_dir.lower() == 'none':
    #     #     output_dir = HydraConfig.get().runtime.output_dir
    #     # else:
    #     #     output_dir = hydra.utils.to_absolute_path(output_dir)
    #     return self._output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, wandb_run=None, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):

        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
            
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)

        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    # if self.world_size > 1:
                    #     model_state_dict = value.module.state_dict()
                    # else:
                    #     model_state_dict = value.state_dict()
                    model_state_dict = {k.replace('module.', ''): v for k, v in value.state_dict().items()}
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(model_state_dict)
                    else:
                        payload['state_dicts'][key] = model_state_dict
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                
                adjusted_state_dict = {k.replace('module.', ''): v for k, v in value.items()}
                self.__dict__[key].load_state_dict(adjusted_state_dict, **kwargs)

                # self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        
        if 'state_dicts' in payload:
            for key in list(payload['state_dicts'].keys()):
                if self.world_size > 1:
                    adapted_dict = adapt_state_dict_for_ddp(payload['state_dicts'][key])
                    payload['state_dicts'][key] = adapted_dict
                else:
                    payload['state_dicts'][key] = payload['state_dicts'][key]

        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
