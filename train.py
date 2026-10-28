"""FP32 full3755 baseline/proposed training using frozen experiment functions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import torch
from project import ROOT, ARCHIVE, modules, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['baseline','proposed'],required=True)
    parser.add_argument('--baseline-run',type=Path,help='The norm_softmax directory produced by train.py --method baseline')
    opts = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA GPU')
    m = modules()
    original = m['helper']
    source = ARCHIVE/'outputs/full3755_cosine_fp32_20261011_230604_699195_seed42/config.json'
    shared = read_json(source)
    args = argparse.Namespace(**shared['args'])
    args.no_amp = True
    rows, configs, hashes, _ = m['full'].load_full_data(ROOT,original)
    if hashes != shared['data_sha256']:
        raise ValueError('Data/split hashes differ from the reported experiment')
    helper = SimpleNamespace(GNTDataset=original.GNTDataset,seed_worker=original.seed_worker,
        make_model=m['full'].make_model,evaluate=original.evaluate,write_json=original.write_json,
        save_checkpoint=original.save_checkpoint,sha256=original.sha256)
    parent = ROOT/'outputs'/f'{opts.method}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    parent.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    m['controls'].seed_everything(args.seed,args.threads)
    probe = helper.make_model()
    initial = m['controls'].state_fingerprint(probe)
    del probe
    torch.cuda.empty_cache()
    reference = None
    if opts.method == 'proposed':
        if opts.baseline_run is None:
            parser.error('--baseline-run is required for the proposed method')
        reference = m['repeats'].warmup_reference(opts.baseline_run)
    mapping = {'global_class_ids':list(range(3755)), 'classes':configs['classes.json']['classes']}
    recipe = {**shared,'args':vars(args),'data_sha256':hashes,'project_root':str(ROOT),
              'entry_point':'train.py','primary_bn_variant':'raw_bn'}
    helper.write_json(parent/'config.json',recipe)
    method = 'norm_softmax' if opts.method == 'baseline' else 'norm_hinge_top1'
    result = m['combo'].train_one(method,parent,args,rows,{'pilot_classes.json':mapping},shared,recipe,
        initial,reference,helper,m['controls'],m['robust'],m['transfer'],m['candidate'],m['repeats'])
    helper.write_json(parent/'result.json',{'completed':True,'results':result})
    print('DONE:',parent)


if __name__ == '__main__':
    main()
