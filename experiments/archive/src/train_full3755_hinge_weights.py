"""Prespecified FP32 full-class auxiliary-weight check: 0.25 / 0.5.

Reuse completed normalized baseline and Top1 weight1, train two new jobs.
Raw BN is primary for every arm. Validation exploration, no official test.
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
NEW_ARMS=(('top1_w025',0.25),('top1_w050',0.5))


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


def scaled_candidate(candidate,weight):
    return SimpleNamespace(forward_features=candidate.forward_features,cosine_scores=candidate.cosine_scores,
        auxiliary_weight=lambda epoch:weight*candidate.auxiliary_weight(epoch))


def write_tables(parent,results,controls):
    endpoints,details=[],[]
    for method,variants in results.items():
        for variant,suites in variants.items():
            require(abs(suites['seen']['clean']-suites['unseen']['clean'])<1e-9,'Clean suite mismatch')
            endpoints.append({'method':method,'bn_variant':variant,'clean_pct':suites['seen']['clean'],
                'mean_of_6_seen_pct':float(np.mean([v for k,v in suites['seen'].items() if k!='clean'])),
                'mean_of_8_unseen_pct':float(np.mean([v for k,v in suites['unseen'].items() if k!='clean'])),
                'lowres_20_pct':suites['unseen']['lowres_20'],'blur_s2.0_pct':suites['unseen']['blur_s2.0']})
            for suite,values in suites.items():
                for condition,value in values.items():
                    details.append({'method':method,'bn_variant':variant,'suite':suite,'condition':condition,
                        'accuracy_pct':value,'delta_vs_norm_softmax_pp':value-results['norm_softmax'][variant][suite][condition],
                        'delta_vs_top1_w1_pp':value-results['top1_w1'][variant][suite][condition]})
    controls.write_csv(parent/'endpoints.csv',endpoints)
    controls.write_csv(parent/'comparison.csv',details)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    opts=parser.parse_args()
    previous=opts.reference_run.resolve()
    require(previous.is_relative_to(ROOT.resolve()),'Reference outside project')
    prior=read_json(previous/'config.json')
    completed=read_json(previous/'result.json')
    require(completed['completed'] and completed['classes']==3755 and completed['seed']==42
            and completed['primary_bn_variant']=='raw_bn','Expected completed full3755 Top1 comparison')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1') and torch.cuda.is_available()
            and torch.cuda.device_count()==1,'Expose one authorized GPU')
    require(str(torch.__version__)==prior['torch'] and str(torchvision.__version__)==prior['torchvision'],
            'Environment changed')
    sources={n:(previous/Path(item['path']).name,item['sha256']) for n,item in prior['source_snapshots'].items()}
    sources['reference_runner']=(previous/'train_full3755_normalized_hinge.py',prior['source_sha256'])
    for n,(path,digest) in sources.items():
        require(sha(path)==digest,f'Changed source snapshot: {n}')
    modules={n:load('hingeweights_'+n,p) for n,(p,_) in sources.items()}
    original,controls,robust,transfer,candidate,repeats,ablation,combination,full,runner=[modules[n] for n in
        ('helper','controls','robust','transfer','candidate','repeats','ablation','combination','full','runner')]
    rows,configs,hashes,_=full.load_full_data(ROOT,original)
    require(hashes==prior['data_sha256'],'Data/splits changed')
    require(len(rows['train'])==748158 and len(rows['val'])==74760,'Unexpected full-class sample counts')
    for filename,expected in (('bn_calibration_full3755_v1.json',prior['bn_calibration']),
                              ('full3755_normalized_hinge_v1.json',prior['protocol']),
                              ('pilot_robustness_v1.json',prior['seen_protocol']),
                              ('pilot_transfer_v1.json',prior['unseen_protocol'])):
        require(read_json(ROOT/'configs'/filename)==expected,f'Changed protocol: {filename}')
    args=argparse.Namespace(**prior['args'])
    require(args.seed==42 and args.epochs==10 and args.batch_size==128 and args.no_amp is True,
            'Expected matched seed42/10epoch/FP32 recipe')
    mapping=prior['evaluation_class_mapping']
    require(mapping['global_class_ids']==list(range(3755)) and mapping['classes']==configs['classes.json']['classes'],
            'Class mapping mismatch')
    runner.verify_index(previous/'val_sample_index.csv',rows['val'],mapping['classes'])
    helper=SimpleNamespace(GNTDataset=original.GNTDataset,seed_worker=original.seed_worker,
        make_model=full.make_model,evaluate=original.evaluate,write_json=original.write_json,
        save_checkpoint=original.save_checkpoint,sha256=original.sha256)
    full_run=Path(prior['full_run']).resolve()
    audit_run=Path(prior['bn_audit_run']).resolve()
    require(all(p.is_relative_to(ROOT.resolve()) for p in (full_run,audit_run)),'Baseline outside project')
    require(sha(previous/'val_sample_index.csv')==sha(audit_run/'val_sample_index.csv')==sha(full_run/'val_sample_index.csv'),
            'Validation indexes differ')
    targets=np.asarray([r[4] for r in rows['val']],dtype=np.int64)
    results={m:{bn:{} for bn in ('raw_bn','calibrated_bn')} for m in ('norm_softmax','top1_w1')}
    provenance=[]
    for method in results:
        old_method='norm_softmax' if method=='norm_softmax' else 'norm_hinge_top1'
        for suite,conditions in (('seen',robust.CONDITIONS),('unseen',transfer.CONDITIONS)):
            for variant in results[method]:
                if method=='top1_w1':
                    path=previous/'norm_hinge_top1'/suite/f'predictions_{variant}.npz'
                elif variant=='raw_bn':
                    path=audit_run/suite/'predictions_norm_softmax.npz'
                else:
                    path=full_run/'norm_softmax'/suite/'predictions_calibrated_bn.npz'
                values=repeats.prediction_metrics(path,targets,[c['name'] for c in conditions],mapping['global_class_ids'])
                require(all(abs(v-completed['results'][old_method][variant][suite][n])<1e-8 for n,v in values.items()),
                        f'Reference accuracy mismatch: {method}/{variant}/{suite}')
                results[method][variant][suite]=values
                provenance.append({'method':method,'suite':suite,'bn_variant':variant,'path':str(path),'sha256':sha(path)})
    parent=ROOT/'outputs'/f'full3755_hinge_weights_{datetime.now():%Y%m%d_%H%M%S_%f}_seed42'
    parent.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT/'logs'/f'{parent.name}.log',encoding='utf-8')])
    logging.info('Output: %s; 2 new FP32 full3755 jobs; auxiliary weights 0.25 and 0.5',parent)
    controls.seed_everything(args.seed,args.threads)
    combination.self_checks(helper,ablation,repeats)
    for arm,weight in NEW_ARMS:
        schedule=[scaled_candidate(candidate,weight).auxiliary_weight(e) for e in range(1,11)]
        require(schedule==[0,weight/2]+[weight]*8,'Weight schedule mismatch')
        logging.info('WEIGHT CHECK PASS %s: %s',arm,schedule)
    protocol={'version':1,'classes':3755,'seed':42,'epochs':10,'precision':'FP32','new_weights':[.25,.5],
        'reused_arms':['norm_softmax','top1_w1'],'auxiliary_schedule':'epoch1=0; epoch2=weight/2; epochs3-10=weight',
        'hinge_margin':.2,'selected':1,'negative_pool':10,'auxiliary_views':'degraded only','normalized_scale':30,
        'checkpoint':'fixed epoch10','primary_bn_variant':'raw_bn for ALL arms','calibrated_bn':'diagnostic only',
        'interpretation':'prespecified limited weight exploration on reused validation data; not confirmatory evidence',
        'data_sha256':hashes,'test_evaluated':False,'confusion_evaluated':False}
    robust.fixed_json(ROOT/'configs/full3755_hinge_weights_v1.json',protocol)
    recipe={**prior,'protocol':protocol,'reference_run':str(previous),'source_sha256':sha(Path(__file__)),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(0),
        'cudnn_enabled':torch.backends.cudnn.enabled,'cudnn_version':torch.backends.cudnn.version(),
        'source_snapshots':{n:{'path':str(p),'sha256':h} for n,(p,h) in sources.items()}}
    helper.write_json(parent/'config.json',recipe)
    helper.write_json(parent/'reused_predictions.json',provenance)
    shutil.copyfile(Path(__file__),parent/Path(__file__).name)
    for path,_ in sources.values():
        shutil.copyfile(path,parent/path.name)
    for filename in ('val_sample_index.csv','classes_full3755.json','initial_state.json'):
        shutil.copyfile(previous/filename,parent/filename)
    write_tables(parent,results,controls)
    initial_hash=read_json(previous/'initial_state.json')['sha256']
    reference=repeats.warmup_reference(full_run/'norm_softmax')
    for arm,weight in NEW_ARMS:
        directory=parent/arm
        directory.mkdir()
        logging.info('START ARM %s: auxiliary_scale=%.2f; output=%s',arm,weight,directory)
        arm_recipe={**recipe,'arm':arm,'auxiliary_scale':weight}
        results[arm]=combination.train_one('norm_hinge_top1',directory,args,rows,{'pilot_classes.json':mapping},
            arm_recipe,arm_recipe,initial_hash,reference,helper,controls,robust,transfer,
            scaled_candidate(candidate,weight),repeats)
        write_tables(parent,results,controls)
        helper.write_json(parent/'progress.json',{'completed':False,'last_completed_arm':arm,'results':results})
        gc.collect()
        torch.cuda.empty_cache()
    helper.write_json(parent/'result.json',{'completed':True,'results':results,'classes':3755,'seed':42,
        'precision':'FP32','primary_bn_variant':'raw_bn','new_training_jobs':2,
        'test_evaluated':False,'confusion_evaluated':False})
    logging.info('DONE: %s',parent)
    print((parent/'endpoints.csv').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':
    main()
