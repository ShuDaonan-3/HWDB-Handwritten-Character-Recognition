"""Repeat the frozen normalized-hinge pilot at seeds 43/44; reuse seed42.

Three methods x three seeds on the same validation writers. Report mean and
sample SD and paired seed differences, not confidence intervals or p values.
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

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torchvision

ROOT=Path('/workspace/shouxieti')
METHODS=('norm_softmax','norm_hinge_top1','norm_hinge_top3')
PAIRS=(('norm_softmax','norm_hinge_top1'),('norm_softmax','norm_hinge_top3'),
       ('norm_hinge_top1','norm_hinge_top3'))
ENDPOINTS=(('seen','clean'),('seen','mean_of_6_seen'),('unseen','mean_of_8_unseen'),
           ('unseen','lowres_20'),('unseen','blur_s2.0'))


def require(value,message):
    if not value:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def endpoint(values,suite,name):
    if name.startswith('mean_of_'):
        condition_values=[v for k,v in values[suite].items() if k!='clean']
        expected=6 if suite=='seen' else 8
        require(len(condition_values)==expected,'Unexpected condition count')
        return float(np.mean(condition_values))
    return values[suite][name]


def write_tables(parent,results,controls):
    per_seed,summary,paired=[],[],[]
    for variant in ('raw_bn','calibrated_bn'):
        for suite,name in ENDPOINTS:
            by_method={}
            for method in METHODS:
                values={seed:endpoint(data[method][variant],suite,name) for seed,data in results.items() if method in data}
                by_method[method]=values
                for seed,value in sorted(values.items()):
                    per_seed.append({'bn_variant':variant,'endpoint':name,'seed':seed,'method':method,'accuracy_pct':value})
                if values:
                    summary.append({'bn_variant':variant,'endpoint':name,'method':method,'n_seeds':len(values),
                        'seeds':';'.join(map(str,sorted(values))),'mean_accuracy_pct':float(np.mean(list(values.values()))),
                        'sample_sd_pp':float(np.std(list(values.values()),ddof=1)) if len(values)>1 else ''})
            for a,b in PAIRS:
                seeds=sorted(set(by_method[a])&set(by_method[b]))
                if seeds:
                    delta=[by_method[b][s]-by_method[a][s] for s in seeds]
                    paired.append({'bn_variant':variant,'endpoint':name,'a':a,'b':b,'n_paired_seeds':len(seeds),
                        'seeds':';'.join(map(str,seeds)),'mean_delta_pp':float(np.mean(delta)),
                        'sample_sd_delta_pp':float(np.std(delta,ddof=1)) if len(delta)>1 else '',
                        'per_seed_delta_pp':';'.join(f'{s}:{d:+.6f}' for s,d in zip(seeds,delta))})
    controls.write_csv(parent/'per_seed.csv',per_seed)
    controls.write_csv(parent/'summary.csv',summary)
    controls.write_csv(parent/'paired_deltas.csv',paired)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    opts=parser.parse_args()
    previous=opts.reference_run.resolve()
    require(previous.is_relative_to(ROOT.resolve()),'Reference outside project')
    prior=read_json(previous/'config.json')
    previous_results=read_json(previous/'result.json')
    require(previous_results['completed'] and previous_results['seed']==42
            and previous_results['primary_bn_variant']=='raw_bn','Expected completed seed42 pilot')
    require(prior['args']['no_amp'] is True,'Reference is not FP32')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1') and torch.cuda.is_available()
            and torch.cuda.device_count()==1,'Expose one authorized GPU')
    require(str(torch.__version__)==prior['torch'] and str(torchvision.__version__)==prior['torchvision'],
            'Use the original environment')
    sources={n:(previous/Path(item['path']).name,item['sha256']) for n,item in prior['source_snapshots'].items()}
    sources['combination']=(previous/'train_pilot_normalized_hinge.py',prior['source_sha256'])
    for name,(path,digest) in sources.items():
        require(sha(path)==digest,f'Snapshot changed: {name}')
    modules={n:load('hingerepeat_'+n,p) for n,(p,_) in sources.items()}
    helper,controls,robust,transfer,candidate,repeats,combination=[modules[n] for n in
        ('helper','controls','robust','transfer','candidate','repeats','combination')]
    require(list(combination.METHODS)==list(METHODS),'Method protocol changed')
    rows,data_config,hashes=helper.load_data(ROOT)
    require(hashes==prior['data_sha256'],'Data/split changed')
    require(len(rows['train'])==99594 and len(rows['val'])==9947 and
            len(data_config['pilot_classes.json']['global_class_ids'])==500,'Expected pilot500')
    for filename,expected in (('pilot_normalized_hinge_v1.json',prior['protocol']),
                              ('bn_calibration_pilot_v1.json',prior['bn_calibration']),
                              ('pilot_robustness_v1.json',prior['seen_protocol']),
                              ('pilot_transfer_v1.json',prior['unseen_protocol'])):
        require(read_json(ROOT/'configs'/filename)==expected,f'Changed protocol: {filename}')
    base_args=argparse.Namespace(**prior['args'])
    require(base_args.seed==42 and base_args.epochs==10 and base_args.batch_size==128,'Unexpected reference recipe')
    targets=np.asarray([r[4] for r in rows['val']],dtype=np.int64)
    class_ids=data_config['pilot_classes.json']['global_class_ids']
    results={42:{}}
    provenance=[]
    for method in METHODS:
        require(read_json(previous/method/'result.json')['completed'],'Incomplete seed42 method')
        require(read_json(previous/method/'config.json')['data_sha256']==hashes,'Seed42 method data changed')
        results[42][method]={bn:{} for bn in ('raw_bn','calibrated_bn')}
        for suite,conditions in (('seen',robust.CONDITIONS),('unseen',transfer.CONDITIONS)):
            for variant in results[42][method]:
                path=previous/method/suite/f'predictions_{variant}.npz'
                values=repeats.prediction_metrics(path,targets,[c['name'] for c in conditions],class_ids)
                reference=previous_results['results'][method][variant][suite]
                require(all(abs(value-reference[name])<1e-8 for name,value in values.items()),'Seed42 accuracy mismatch')
                results[42][method][variant][suite]=values
                provenance.append({'method':method,'suite':suite,'bn_variant':variant,'path':str(path),'sha256':sha(path)})
        require(abs(results[42][method]['raw_bn']['seen']['clean']-
                    results[42][method]['raw_bn']['unseen']['clean'])<1e-9,'Clean suite mismatch')
    parent=ROOT/'outputs'/f'pilot_normalized_hinge_repeats_{datetime.now():%Y%m%d_%H%M%S_%f}'
    parent.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT/'logs'/f'{parent.name}.log',encoding='utf-8')])
    logging.info('Output: %s; 6 new FP32 jobs: 3 methods x seeds43/44; reuse seed42',parent)
    controls.seed_everything(42,base_args.threads)
    combination.self_checks(helper,modules['ablation'],repeats)
    protocol={'version':1,'seeds':[42,43,44],'methods':list(METHODS),'precision':'FP32',
        'primary_bn_variant':'raw_bn','reuse_seed42':True,'new_training_jobs':6,
        'method_protocol':prior['protocol'],'split':'same fixed writers/classes for every seed',
        'selection':'fixed epoch10; report all methods and seeds',
        'interpretation':'mean and sample SD over training seeds on the SAME validation images; not confidence intervals',
        'test_evaluated':False,'confusion_evaluated':False}
    robust.fixed_json(ROOT/'configs/pilot_normalized_hinge_repeats_v1.json',protocol)
    recipe={**prior,'reference_run':str(previous),'protocol':protocol,'source_sha256':sha(Path(__file__)),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(0),
        'source_snapshots':{n:{'path':str(p),'sha256':h} for n,(p,h) in sources.items()}}
    helper.write_json(parent/'config.json',recipe)
    helper.write_json(parent/'reused_predictions.json',provenance)
    shutil.copyfile(Path(__file__),parent/Path(__file__).name)
    for path,_ in sources.values():
        shutil.copyfile(path,parent/path.name)
    shutil.copyfile(previous/'val_sample_index.csv',parent/'val_sample_index.csv')
    write_tables(parent,results,controls)
    for seed in (43,44):
        args=argparse.Namespace(**{**vars(base_args),'seed':seed})
        seed_dir=parent/f'seed{seed}'
        seed_dir.mkdir()
        controls.seed_everything(seed,args.threads)
        probe=helper.make_model()
        initial_hash=controls.state_fingerprint(probe)
        del probe
        gc.collect()
        torch.cuda.empty_cache()
        helper.write_json(seed_dir/'initial_state.json',{'seed':seed,'sha256':initial_hash})
        seed_recipe={**recipe,'args':vars(args),'augmentation_seed':seed+3000}
        helper.write_json(seed_dir/'config.json',seed_recipe)
        results[seed]={}
        reference=None
        for method in METHODS:
            results[seed][method]=combination.train_one(method,seed_dir,args,rows,data_config,prior,seed_recipe,
                initial_hash,reference,helper,controls,robust,transfer,candidate,repeats)
            if method=='norm_softmax':
                reference=repeats.warmup_reference(seed_dir/method)
            write_tables(parent,results,controls)
            helper.write_json(parent/'progress.json',{'results':results,'completed':False,'current_seed':seed,
                'last_completed_method':method})
            gc.collect()
            torch.cuda.empty_cache()
    helper.write_json(parent/'result.json',{'completed':True,'results':results,'seeds':[42,43,44],
        'new_training_jobs':6,'primary_bn_variant':'raw_bn','test_evaluated':False,'confusion_evaluated':False})
    logging.info('DONE: %s',parent)
    print('PRIMARY RAW-BN RESULTS:',flush=True)
    for filename in ('summary.csv','paired_deltas.csv'):
        with (parent/filename).open(encoding='utf-8',newline='') as f:
            rows_out=list(csv.DictReader(f))
        print(','.join(rows_out[0]),flush=True)
        for row in rows_out:
            if row['bn_variant']=='raw_bn':
                print(','.join(row.values()),flush=True)


if __name__=='__main__':
    main()
