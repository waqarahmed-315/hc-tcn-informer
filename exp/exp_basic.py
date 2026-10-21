import os, json, random, platform
import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

class Exp_Basic:
    def __init__(self,args):
        self.args=args; set_random_seed(args.seed)
        self.run_dir=os.path.join(args.results_root,"runs",args.setting); os.makedirs(self.run_dir,exist_ok=True)
        with open(os.path.join(self.run_dir,"args.json"),"w") as f: json.dump(vars(args),f,indent=2)
        self.device=self._acquire_device(); self.model=self._build_model().to(self.device)
    def _build_model(self): raise NotImplementedError
    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device=torch.device(f"cuda:{self.args.gpu}")
            print("GPU:",torch.cuda.get_device_name(device)); return device
        print("Using CPU"); return torch.device("cpu")
