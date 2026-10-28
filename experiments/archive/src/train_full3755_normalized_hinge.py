"""3755-class FP32 Top1 combination scale check, with reused matched baseline.

One new ten-epoch seed42 training job. Primary comparison uses raw BN for
both methods. Calibrated BN is diagnostic. No test/confusion evaluation.
"""
import argparse
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

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torchvision

ROOT=Path('/workspace/shouxieti')


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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats-run',type=Path,required=True)
    parser.add_argument('--full-run',type=Path,required=True)
    parser.add_argument('--bn-audit-run',type=Path,required=True)
    opts=parser.parse_args()
    repeats_run,full_run,audit_run=[p.resolve() for p in (opts.repeats_run,opts.full_run,opts.bn_audit_run)]
    require(all(p.is_relative_to(ROOT.resolve()) for p in (repeats_run,full_run,audit_run)),'Reference outside project')
    pilot=read_json(repeats_run/'config.json')
    prior=read_json(full_run/'config.json')
    audit=read_json(audit_run/'result.json')
    require(read_json(repeats_run/'result.json')['completed'] and read_json(full_run/'result.json')['completed']
            and audit['completed'],'Incomplete references')
    require(Path(audit['run_dir']).resolve()==full_run,'BN audit refers to a different training run')
    require(pilot['protocol']['primary_bn_variant']=='raw_bn' and prior['args']['no_amp'] is True,
            'Expected FP32/raw-BN references')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1') and torch.cuda.is_available()
            and torch.cuda.device_count()==1,'Expose one authorized GPU')
    require(str(torch.__version__)==prior['torch']==pilot['torch'] and
            str(torchvision.__version__)==prior['torchvision']==pilot['torchvision'],'Environment changed')
    sources={n:(repeats_run/Path(item['path']).name,item['sha256']) for n,item in pilot['source_snapshots'].items()}
    for n in ('helper','controls','robust','transfer','candidate','repeats','ablation'):
        require(sources[n][1]==prior['source_snapshots'][n]['sha256'],f'Pilot/full implementations differ: {n}')
    item=prior['source_snapshots']['full']
    sources['full']=(full_run/Path(item['path']).name,item['sha256'])
    sources['runner']=(full_run/'train_full3755_cosine.py',prior['source_sha256'])
    for n,(path,digest) in sources.items():
        require(sha(path)==digest,f'Source snapshot changed: {n}')
    modules={n:load('fullhinge_'+n,p) for n,(p,_) in sources.items()}
    original,controls,robust,transfer,candidate,repeats,ablation,combination,full,runner=[modules[n] for n in
        ('helper','controls','robust','transfer','candidate','repeats','ablation','combination','full','runner')]
    require(combination.METHODS['norm_hinge_top1']=={'head':'normalized','margin':0.0,'selected':1},
            'Top1 method definition changed')
    rows,configs,hashes,_=full.load_full_data(ROOT,original)
    require(hashes==prior['data_sha256']==pilot['data_sha256'],'Data/splits changed')
    require(len(rows['train'])==748158 and len(rows['val'])==74760,'Unexpected full-class counts')
    for filename,expected in (('bn_calibration_full3755_v1.json',prior['bn_calibration']),
                              ('pilot_robustness_v1.json',prior['seen_protocol']),
                              ('pilot_transfer_v1.json',prior['unseen_protocol']),
                              ('pilot_normalized_hinge_repeats_v1.json',pilot['protocol'])):
        require(read_json(ROOT/'configs'/filename)==expected,f'Protocol changed: {filename}')
    require(prior['seen_protocol']==pilot['seen_protocol'] and prior['unseen_protocol']==pilot['unseen_protocol'],
            'Condition protocols differ')
    args=argparse.Namespace(**prior['args'])
    require(args.seed==42 and args.epochs==10 and args.batch_size==128 and args.no_amp,'Expected seed42/FP32/10epochs/B128')
    for key in ('lr','weight_decay','threads','no_amp'):
        require(vars(args)[key]==pilot['args'][key],f'Pilot/full recipe differs: {key}')
    mapping=prior['evaluation_class_mapping']
    require(mapping['global_class_ids']==list(range(3755)) and mapping['classes']==configs['classes.json']['classes'],
            'Full class mapping changed')
    runner.verify_index(full_run/'val_sample_index.csv',rows['val'],mapping['classes'])
    require(sha(full_run/'val_sample_index.csv')==sha(audit_run/'val_sample_index.csv'),'BN audit sample order changed')
    helper=SimpleNamespace(GNTDataset=original.GNTDataset,seed_worker=original.seed_worker,
        make_model=full.make_model,evaluate=original.evaluate,write_json=original.write_json,
        save_checkpoint=original.save_checkpoint,sha256=original.sha256)
    parent=ROOT/'outputs'/f'full3755_normalized_hinge_{datetime.now():%Y%m%d_%H%M%S_%f}_seed42'
    parent.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT/'logs'/f'{parent.name}.log',encoding='utf-8')])
    logging.info('Output: %s; one new FP32 Top1 job; train=%d val=%d; GPU=%s',parent,len(rows['train']),len(rows['val']),torch.cuda.get_device_name(0))
    controls.seed_everything(args.seed,args.threads)
    combination.self_checks(helper,ablation,repeats)
    protocol={'version':1,'classes':3755,'seed':42,'precision':'FP32','epochs':10,
        'methods':['norm_softmax','norm_hinge_top1'],'new_training_jobs':1,'reused_baseline':'completed full3755 FP32 norm_softmax epoch10',
        'normalized_scale':30.0,'hinge_margin':0.2,'negative_pool':10,'selected':1,
        'auxiliary_views':'degraded only','auxiliary_weight':'epoch1=0; epoch2=0.5; epochs3-10=1',
        'primary_bn_variant':'raw_bn for both methods','calibrated_bn':'diagnostic only',
        'checkpoint':'fixed epoch10; no best epoch selection','bn_calibration':prior['bn_calibration'],
        'data_sha256':hashes,'interpretation':'pilot-selected candidate; single-seed scale check on reused validation writers',
        'test_evaluated':False,'confusion_evaluated':False}
    robust.fixed_json(ROOT/'configs/full3755_normalized_hinge_v1.json',protocol)
    recipe={**prior,'protocol':protocol,'repeats_run':str(repeats_run),'full_run':str(full_run),'bn_audit_run':str(audit_run),
        'source_sha256':sha(Path(__file__)),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu':torch.cuda.get_device_name(0),'cudnn_enabled':torch.backends.cudnn.enabled,
        'cudnn_version':torch.backends.cudnn.version(),
        'source_snapshots':{n:{'path':str(p),'sha256':h} for n,(p,h) in sources.items()}}
    helper.write_json(parent/'config.json',recipe)
    shutil.copyfile(Path(__file__),parent/Path(__file__).name)
    for path,_ in sources.values():
        shutil.copyfile(path,parent/path.name)
    for filename in ('classes_full3755.json','val_sample_index.csv','initial_state.json'):
        shutil.copyfile(full_run/filename,parent/filename)
    baseline={'raw_bn':{},'calibrated_bn':{}}
    targets=np.asarray([r[4] for r in rows['val']],dtype=np.int64)
    reused=[]
    for suite,conditions in (('seen',robust.CONDITIONS),('unseen',transfer.CONDITIONS)):
        for variant in baseline:
            path=audit_run/suite/'predictions_norm_softmax.npz' if variant=='raw_bn' else full_run/'norm_softmax'/suite/'predictions_calibrated_bn.npz'
            values=repeats.prediction_metrics(path,targets,[c['name'] for c in conditions],mapping['global_class_ids'])
            require(all(abs(v-audit['results']['norm_softmax'][variant][suite][n])<1e-8 for n,v in values.items()),
                    'Reused baseline accuracy mismatch')
            baseline[variant][suite]=values
            reused.append({'suite':suite,'bn_variant':variant,'path':str(path),'sha256':sha(path)})
    helper.write_json(parent/'reused_predictions.json',reused)
    results={'norm_softmax':baseline}
    combination.write_tables(parent,results,controls)
    initial_hash=read_json(full_run/'initial_state.json')['sha256']
    reference=repeats.warmup_reference(full_run/'norm_softmax')
    logging.info('DATA/PROVENANCE CHECKS PASS; normalized baseline reproduced; matched full-class BN protocol')
    results['norm_hinge_top1']=combination.train_one('norm_hinge_top1',parent,args,rows,{'pilot_classes.json':mapping},
        recipe,recipe,initial_hash,reference,helper,controls,robust,transfer,candidate,repeats)
    combination.write_tables(parent,results,controls)
    helper.write_json(parent/'result.json',{'completed':True,'results':results,'classes':3755,'seed':42,
        'precision':'FP32','primary_bn_variant':'raw_bn','new_training_jobs':1,
        'test_evaluated':False,'confusion_evaluated':False})
    gc.collect()
    logging.info('DONE: %s',parent)
    print((parent/'endpoints.csv').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':
    main()
