"""Full-class matched-budget normalized-softmax/CosFace validation comparison.

Import verified snapshots from completed full-class and pilot ablation runs.
Default: two new jobs and reused baselines. --fp32: four matched FP32 jobs.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torchvision

ROOT = Path('/workspace/shouxieti')
METHODS = ('norm_softmax', 'cosface_m010')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_index(path, rows, classes):
    with path.open(encoding='utf-8', newline='') as stream:
        count = 0
        for count, item in enumerate(csv.DictReader(stream), 1):
            require(count <= len(rows), 'Extra validation index row')
            p, offset, width, height, label = rows[count-1]
            expected = (count-1, Path(p).relative_to(ROOT).as_posix(), offset, width, height,
                        int(Path(p).stem.split('-')[0]), label, label, classes[label])
            actual = (int(item['row']), item['gnt_path'], int(item['pixel_offset']),
                      int(item['width']), int(item['height']), int(item['writer_id']),
                      int(item['local_class_id']), int(item['global_class_id']), item['character'])
            require(actual == expected, f'Validation row mismatch: {count}')
        require(count == len(rows), 'Missing validation index rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-run', required=True, type=Path)
    parser.add_argument('--ablations-run', required=True, type=Path)
    parser.add_argument('--fp32', action='store_true',
                        help='Train all four methods in FP32; rerun both baselines for a matched comparison')
    opts = parser.parse_args()
    full_run, pilot_run = opts.full_run.resolve(), opts.ablations_run.resolve()
    require(all(p.is_relative_to(ROOT.resolve()) for p in (full_run, pilot_run)), 'Run outside project')
    prior, pilot = read_json(full_run/'config.json'), read_json(pilot_run/'config.json')
    require(read_json(full_run/'result.json')['completed'] and read_json(pilot_run/'result.json')['completed'],
            'Reference runs must be completed')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '1'),
            'Set CUDA_VISIBLE_DEVICES to one authorized GPU: 0 or 1')
    require(torch.cuda.is_available(), 'CUDA unavailable; check container GPU access and PyTorch environment')
    require(torch.cuda.device_count() == 1, 'Exactly one visible GPU is required')
    require(str(torch.__version__) == prior['torch'] == pilot['torch'] and
            str(torchvision.__version__) == prior['torchvision'] == pilot['torchvision'], 'Environment changed')
    sources = {name: (full_run/Path(item['path']).name, item['sha256'])
               for name,item in prior['source_snapshots'].items()}
    for name, (_, expected) in sources.items():
        require(pilot['source_snapshots'][name]['sha256'] == expected, f'Pilot/full implementations differ: {name}')
    sources['full'] = (full_run/'train_full3755_margin.py', prior['source_sha256'])
    sources['ablation'] = (pilot_run/'train_pilot_margin_ablations.py', pilot['source_sha256'])
    for name,(path,expected) in sources.items():
        require(digest(path) == expected, f'Source changed: {name}')
    modules = {name:load_module(f'fullcos_{name}', path) for name,(path,_) in sources.items()}
    original, controls, robust, transfer, candidate, repeats, full, ablation = [modules[n] for n in
        ('helper','controls','robust','transfer','candidate','repeats','full','ablation')]
    rows, configs, hashes, _ = full.load_full_data(ROOT, original)
    require(hashes == prior['data_sha256'] == pilot['data_sha256'], 'Data or splits changed')
    require(len(rows['train']) == 748158 and len(rows['val']) == 74760, 'Unexpected full-class counts')
    for filename,expected in (('full3755_margin_v1.json', prior['protocol']),
                              ('bn_calibration_full3755_v1.json', prior['bn_calibration']),
                              ('pilot_robustness_v1.json', prior['seen_protocol']),
                              ('pilot_transfer_v1.json', prior['unseen_protocol'])):
        require(read_json(ROOT/'configs'/filename) == expected, f'Protocol changed: {filename}')
    require(prior['seen_protocol'] == pilot['seen_protocol'] and prior['unseen_protocol'] == pilot['unseen_protocol'],
            'Evaluation protocols differ')
    args = argparse.Namespace(**prior['args'])
    require(args.seed == 42 and args.epochs == 10 and args.batch_size == 128, 'Unexpected training budget')
    for key in ('lr','weight_decay','no_amp','threads'):
        require(vars(args)[key] == pilot['args'][key], f'Pilot/full setting differs: {key}')
    if opts.fp32:
        args.no_amp = True
    mapping = {'global_class_ids':list(range(3755)), 'classes':configs['classes.json']['classes']}
    require(mapping == prior['evaluation_class_mapping'], 'Class mapping changed')
    verify_index(full_run/'val_sample_index.csv', rows['val'], mapping['classes'])
    evaluation_data = {'pilot_classes.json':mapping}
    helper = SimpleNamespace(GNTDataset=original.GNTDataset, seed_worker=original.seed_worker,
        make_model=full.make_model, evaluate=original.evaluate, write_json=original.write_json,
        save_checkpoint=original.save_checkpoint, sha256=original.sha256)
    stage = 'full3755_cosine_fp32' if opts.fp32 else 'full3755_cosine'
    new_jobs = 4 if opts.fp32 else 2
    parent = ROOT/'outputs'/f'{stage}_{datetime.now():%Y%m%d_%H%M%S_%f}_seed42'
    parent.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT/'logs'/f'{parent.name}.log', encoding='utf-8')])
    logging.info('Output: %s; %d new jobs; train=%d val=%d; GPU=%s; no_amp=%s',parent,new_jobs,len(rows['train']),len(rows['val']),torch.cuda.get_device_name(0),args.no_amp)
    controls.seed_everything(args.seed,args.threads)
    ablation.self_checks(helper,repeats)
    if opts.fp32:
        # Verify several real optimizer steps before committing to four long runs.
        controls.seed_everything(args.seed,args.threads)
        probe = helper.make_model().train()
        optimizer = torch.optim.AdamW(probe.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        loader = controls.make_loader(helper,rows['train'],128,args.workers,True,args.seed)
        generator = torch.Generator().manual_seed(args.seed+3000)
        import torch.nn.functional as F
        for step,(images,labels) in enumerate(loader,1):
            images,labels = images.cuda(),labels.cuda()
            second,_ = controls.second_view(images,generator,robust)
            optimizer.zero_grad(set_to_none=True)
            unused_affine,features = candidate.forward_features(probe,torch.cat((images,second)))
            loss = F.cross_entropy(ablation.margin_logits(candidate.cosine_scores(probe,features),
                torch.cat((labels,labels)),.1),torch.cat((labels,labels)))
            require(bool(torch.isfinite(loss)),f'FP32 preflight non-finite loss at step {step}')
            loss.backward()
            require(all(bool(torch.isfinite(p.grad).all()) for p in probe.parameters() if p.grad is not None),
                    f'FP32 preflight non-finite gradient at step {step}')
            optimizer.step()
            require(all(bool(torch.isfinite(p).all()) for p in probe.parameters()),'FP32 preflight invalid weights')
            logging.info('FP32 PREFLIGHT step=%d/5 loss=%.4f',step,loss.item())
            if step == 5:
                break
        del probe,optimizer,loader,loss,features,unused_affine,images,labels,second
        gc.collect()
        torch.cuda.empty_cache()
        logging.info('FP32 PREFLIGHT PASS: 5 optimizer steps; actual runs restart from frozen initialization')
    protocol = {'version':1,'classes':3755,'seed':42,'epochs':10,'new_methods':list(METHODS),
        'method_specs':{m:ablation.METHODS[m] for m in METHODS},'normalized_scale':30,
        'inference':'normalized cosine logits without ground-truth margin or bias',
        'checkpoint_selection':'fixed epoch 10; no validation-best selection',
        'initialization':'same random initial state as completed full-class baseline',
        'bn_calibration':prior['bn_calibration'], 'data_sha256':hashes,
        'interpretation':'single-seed validation comparison; pilot-selected baselines; no convergence or novelty claim',
        'test_evaluated':False,'confusion_evaluated':False}
    if opts.fp32:
        protocol.update(new_methods=['aug_ce','margin_fixed',*METHODS],precision='FP32',
                        baselines='retrained in FP32; old AMP results excluded from comparison tables')
    robust.fixed_json(ROOT/'configs'/f'{stage}_v1.json',protocol)
    recipe = {**prior,'args':vars(args),'protocol':protocol,'full_run':str(full_run),'ablations_run':str(pilot_run),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'precision':'FP32' if args.no_amp else 'AMP',
        'cudnn_enabled':torch.backends.cudnn.enabled,'cudnn_version':torch.backends.cudnn.version(),
        'source_sha256':digest(Path(__file__)),
        'source_snapshots':{n:{'path':str(p),'sha256':h} for n,(p,h) in sources.items()}}
    helper.write_json(parent/'config.json',recipe)
    shutil.copyfile(Path(__file__),parent/Path(__file__).name)
    for path,_ in sources.values():
        shutil.copyfile(path,parent/path.name)
    for filename in ('val_sample_index.csv','classes_full3755.json','initial_state.json'):
        shutil.copyfile(full_run/filename,parent/filename)
    targets = np.asarray([r[4] for r in rows['val']],dtype=np.int64)
    results = {'aug_ce':{},'margin_fixed':{}}
    reused = []
    for method in (() if opts.fp32 else tuple(results)):
        require(read_json(full_run/method/'config.json')['data_sha256'] == hashes, 'Reference method data changed')
        for suite,conditions in (('seen',robust.CONDITIONS),('unseen',transfer.CONDITIONS)):
            directory = full_run/method if method == 'aug_ce' and suite == 'seen' else full_run/method/suite
            path = directory/'predictions_calibrated_bn.npz'
            results[method][suite] = repeats.prediction_metrics(path,targets,[c['name'] for c in conditions],mapping['global_class_ids'])
            reused.append({'method':method,'suite':suite,'path':str(path),'sha256':digest(path)})
    helper.write_json(parent/'reused_predictions.json',reused)
    initial_hash = read_json(full_run/'initial_state.json')['sha256']
    if opts.fp32:
        logging.info('PROVENANCE/DATA CHECKS PASS; starting matched FP32 baselines; frozen BN calibration reused')
        controls.train_one('aug_ce',parent,args,rows,evaluation_data,prior['bn_calibration'],
                           prior['seen_protocol'],helper,robust,recipe)
        checkpoint = torch.load(parent/'aug_ce/epoch10_bn_calibrated.pt',map_location='cpu',weights_only=False)
        baseline = helper.make_model()
        baseline.load_state_dict(checkpoint['model'],strict=True)
        del checkpoint
        results['aug_ce'] = repeats.evaluate_final(baseline,parent/'aug_ce',rows,args,evaluation_data,
                                                  helper,controls,robust,transfer)
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
        reference = repeats.warmup_reference(parent/'aug_ce')
        results['margin_fixed'] = repeats.train_fixed(parent,args,rows,evaluation_data,recipe,recipe,
            initial_hash,reference,helper,controls,robust,transfer,candidate)
        gc.collect()
        torch.cuda.empty_cache()
    else:
        reference = repeats.warmup_reference(full_run/'aug_ce')
        logging.info('PROVENANCE/DATA CHECKS PASS; reference predictions reproduced; frozen full-class BN calibration reused')
    ablation.write_tables(parent,results,controls)
    for method in METHODS:
        results[method] = ablation.train_one(method,parent,args,rows,evaluation_data,recipe,recipe,
            initial_hash,reference,helper,controls,robust,transfer,candidate,repeats)
        ablation.write_tables(parent,results,controls)
        gc.collect()
        torch.cuda.empty_cache()
    helper.write_json(parent/'result.json',{'completed':True,'classes':3755,'seed':42,'results':results,
        'new_training_jobs':new_jobs,'precision':recipe['precision'],'test_evaluated':False,'confusion_evaluated':False})
    logging.info('DONE: %s',parent)
    print((parent/'endpoints.csv').read_text(encoding='utf-8'),flush=True)


if __name__ == '__main__':
    main()
