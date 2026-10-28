"""Read-only first-training-batch diagnosis; no optimizer steps or checkpoint changes."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F

ROOT = Path('/workspace/shouxieti')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stats(x):
    x = x.detach().float()
    finite = torch.isfinite(x)
    good = x[finite]
    return {'shape':list(x.shape), 'finite':bool(finite.all()),
            'nonfinite':int((~finite).sum()),
            'min':float(good.min()) if good.numel() else None,
            'max':float(good.max()) if good.numel() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-run',type=Path,required=True)
    parser.add_argument('--disable-cudnn', action='store_true',
                        help='Diagnostic only: compare using the alternative convolution backend')
    opts = parser.parse_args()
    run = opts.failed_run.resolve()
    assert run.is_relative_to(ROOT.resolve()), 'Run outside project'
    config = json.loads((run/'config.json').read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1'), 'Select one allocated GPU'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'CUDA unavailable or multiple GPUs visible'
    modules = {}
    for name,item in config['source_snapshots'].items():
        path = run/Path(item['path']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'], f'Snapshot changed: {name}'
        modules[name] = load('diagcos_'+name,path)
    helper,controls,robust,candidate,full,ablation = [modules[n] for n in
        ('helper','controls','robust','candidate','full','ablation')]
    args = argparse.Namespace(**config['args'])
    rows,_,hashes,_ = full.load_full_data(ROOT,helper)
    assert hashes==config['data_sha256'], 'Data changed'
    controls.seed_everything(args.seed,args.threads)
    if opts.disable_cudnn:
        torch.backends.cudnn.enabled = False
    model = full.make_model()
    initial_hash = json.loads((run/'initial_state.json').read_text())['sha256']
    assert controls.state_fingerprint(model)==initial_hash, 'Initialization changed'
    model.fc.bias.requires_grad_(False)
    loader = controls.make_loader(helper,rows['train'],128,args.workers,True,args.seed)
    images,labels = next(iter(loader))
    images,labels = images.cuda(),labels.cuda()
    generator = torch.Generator().manual_seed(args.seed+3000)
    second,counts = controls.second_view(images,generator,robust)
    inputs = torch.cat((images,second))
    targets = torch.cat((labels,labels))
    report = {'gpu':torch.cuda.get_device_name(0),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'torch':str(torch.__version__),'cuda':torch.version.cuda,'failed_run':str(run),
              'cudnn_enabled':torch.backends.cudnn.enabled,'cudnn_version':torch.backends.cudnn.version(),
              'initial_state_sha256':initial_hash,'images':stats(images),'second_view':stats(second),
              'labels_min':int(labels.min()),'labels_max':int(labels.max()),
              'augmentation_counts':counts.tolist(),'trials':[]}
    print('INPUT:',json.dumps({k:v for k,v in report.items() if k!='trials'}),flush=True)
    initial = copy.deepcopy(model.state_dict())
    # Every forward starts with identical weights AND BN buffers, and the same inputs.
    for amp in (True,False):
        for repeat in (1,2):
            model.load_state_dict(initial)
            model.train()
            model.zero_grad(set_to_none=True)
            trial={'amp':amp,'repeat':repeat,'first_nonfinite_module':None}
            handles=[]
            def hook(name):
                def record(module, ins, out):
                    if isinstance(out,torch.Tensor) and trial['first_nonfinite_module'] is None:
                        if not bool(torch.isfinite(out).all()):
                            trial['first_nonfinite_module']={'name':name,'output':stats(out)}
                return record
            for name,module in model.named_modules():
                if name and not list(module.children()):
                    handles.append(module.register_forward_hook(hook(name)))
            try:
                with torch.cuda.amp.autocast(enabled=amp):
                    affine,features=candidate.forward_features(model,inputs)
                scores=candidate.cosine_scores(model,features)
                logits=ablation.margin_logits(scores,targets,0.0)
                loss=F.cross_entropy(logits,targets)
                trial.update(features=stats(features),affine_logits=stats(affine),
                             cosine_scores=stats(scores),normalized_logits=stats(logits),
                             loss=float(loss) if bool(torch.isfinite(loss)) else None,
                             loss_finite=bool(torch.isfinite(loss)))
                if trial['loss_finite']:
                    # Unscaled diagnostic gradient, not an AMP optimizer/scaler test.
                    loss.backward()
                    trial['nonfinite_gradient_parameters']=[n for n,p in model.named_parameters()
                        if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            except Exception as exc:
                trial['exception']=f'{type(exc).__name__}: {exc}'
            finally:
                for handle in handles:
                    handle.remove()
            report['trials'].append(trial)
            print('TRIAL:',json.dumps(trial),flush=True)
    out=ROOT/'outputs'/f'diagnose_fullcos_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('DONE:',out,flush=True)


if __name__=='__main__':
    main()
