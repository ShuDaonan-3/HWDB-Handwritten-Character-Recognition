"""Portable entry points around the archived, experiment-tested implementation."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'experiments/archive'


def load(name, path):
    spec = importlib.util.spec_from_file_location('hccr_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    names = {'helper':'train_pilot.py', 'controls':'train_pilot_controls.py',
             'robust':'eval_pilot_robustness.py', 'transfer':'eval_pilot_transfer.py',
             'candidate':'train_pilot_candidates.py', 'repeats':'train_pilot_margin_repeats.py',
             'full':'train_full3755_margin.py', 'combo':'train_pilot_normalized_hinge.py'}
    return {name:load(name, ARCHIVE/'src'/filename) for name,filename in names.items()}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_checkpoint(path, device, m):
    import torch
    from torchvision.models import resnet18
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(1,64,3,stride=1,padding=1,bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512,3755)
    model.load_state_dict(checkpoint['model'],strict=True)
    if checkpoint['epoch'] != 10:
        raise ValueError('Expected epoch-10 checkpoint')
    return m['combo'].InferenceHead(model,True).to(device).eval()
