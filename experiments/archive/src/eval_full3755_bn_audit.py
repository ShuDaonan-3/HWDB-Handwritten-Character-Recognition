"""Compare saved raw and calibrated epoch-10 checkpoints. Evaluation only."""
import argparse
import csv
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

ROOT = Path('/workspace/shouxieti')
METHODS = ('aug_ce','margin_fixed','norm_softmax','cosface_m010')


def require(condition,message):
    if not condition:
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


def endpoint_rows(results):
    rows=[]
    for method,variants in results.items():
        for variant,suites in variants.items():
            require(abs(suites['seen']['clean']-suites['unseen']['clean'])<1e-9,'Clean suite mismatch')
            rows.append({'method':method,'bn_variant':variant,'clean_pct':suites['seen']['clean'],
                'mean_of_6_seen_pct':float(np.mean([v for k,v in suites['seen'].items() if k!='clean'])),
                'mean_of_8_unseen_pct':float(np.mean([v for k,v in suites['unseen'].items() if k!='clean'])),
                'lowres_20_pct':suites['unseen']['lowres_20'],'blur_s2.0_pct':suites['unseen']['blur_s2.0']})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    opts=parser.parse_args()
    run=opts.run_dir.resolve()
    require(run.is_relative_to(ROOT.resolve()),'Run outside project')
    config=read_json(run/'config.json')
    require(read_json(run/'result.json')['completed'],'Training run incomplete')
    require(config['args']['no_amp'] is True,'Expected completed FP32 comparison')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1') and torch.cuda.is_available()
            and torch.cuda.device_count()==1,'Expose one authorized GPU')
    require(str(torch.__version__)==config['torch'],'PyTorch version changed')
    sources={name:(run/Path(item['path']).name,item['sha256'])
             for name,item in config['source_snapshots'].items()}
    sources['runner']=(run/'train_full3755_cosine.py',config['source_sha256'])
    for name,(path,digest) in sources.items():
        require(sha(path)==digest,f'Changed snapshot: {name}')
    modules={n:load('bnaudit_'+n,p) for n,(p,_) in sources.items()}
    original,controls,robust,transfer,repeats,full,ablation,runner=[modules[n] for n in
        ('helper','controls','robust','transfer','repeats','full','ablation','runner')]
    args=argparse.Namespace(**config['args'])
    controls.seed_everything(args.seed,args.threads)
    rows,configs,hashes,_=full.load_full_data(ROOT,original)
    require(hashes==config['data_sha256'],'Data or split changed')
    mapping=config['evaluation_class_mapping']
    require(mapping['classes']==configs['classes.json']['classes'],'Class mapping changed')
    runner.verify_index(run/'val_sample_index.csv',rows['val'],mapping['classes'])
    helper=SimpleNamespace(GNTDataset=original.GNTDataset,seed_worker=original.seed_worker,
                           make_model=full.make_model,sha256=original.sha256)
    out=ROOT/'outputs'/f'full3755_bn_audit_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT/'logs'/f'{out.name}.log',encoding='utf-8')])
    logging.info('Output: %s; evaluate four raw checkpoints; reuse calibrated predictions; no training',out)
    models,results,checkpoint_audit={}, {}, []
    for method in METHODS:
        rawpath=run/method/'last.pt'
        calpath=run/method/'epoch10_bn_calibrated.pt'
        raw=torch.load(rawpath,map_location='cpu',weights_only=False)
        calibrated=torch.load(calpath,map_location='cpu',weights_only=False)
        require(raw['epoch']==calibrated['epoch']==10,'Wrong checkpoint epoch')
        require(raw['config']==calibrated['config']==read_json(run/method/'config.json'),'Checkpoint metadata mismatch')
        require(calibrated['bn_calibration']==config['bn_calibration'],'Calibration config mismatch')
        base=full.make_model()
        base.load_state_dict(raw['model'],strict=True)
        parameter_names=set(dict(base.named_parameters()))
        require(set(raw['model'])==set(calibrated['model']),'Checkpoint keys differ')
        changed=[name for name,value in raw['model'].items() if not torch.equal(value,calibrated['model'][name])]
        require(not parameter_names.intersection(changed),'Calibration changed learned parameters')
        require(all(name.endswith(('running_mean','running_var','num_batches_tracked')) for name in changed),
                'Unexpected calibration state change')
        normalized=method in ('norm_softmax','cosface_m010')
        if normalized:
            require(raw['config']['inference_head']=='normalized' and raw['config']['inference_scale']==30,
                    'Normalized inference config mismatch')
        models[method]=ablation.InferenceHead(base,True).eval() if normalized else base.eval()
        results[method]={'raw_bn':{},'calibrated_bn':{}}
        checkpoint_audit.append({'method':method,'raw_sha256':sha(rawpath),'calibrated_sha256':sha(calpath),
            'changed_buffers':changed,'learned_parameters_identical':True,'saved_raw_val_accuracy_pct':100*raw['val_accuracy']})
        del raw,calibrated
    targets=np.asarray([r[4] for r in rows['val']],dtype=np.int64)
    loader=controls.make_loader(helper,rows['val'],128,args.workers,False,args.seed+1)
    for suite,adapter in (('seen',robust),('unseen',transfer.ProtocolAdapter(robust.corrupt))):
        directory=out/suite
        directory.mkdir()
        values=controls.evaluate_conditions(models,loader,targets,adapter,directory,{'pilot_classes.json':mapping})
        for method in METHODS:
            results[method]['raw_bn'][suite]={r['condition']:r['accuracy_pct'] for r in values if r['bn_variant']==method}
            olddir=run/method if method=='aug_ce' and suite=='seen' else run/method/suite
            results[method]['calibrated_bn'][suite]=repeats.prediction_metrics(olddir/'predictions_calibrated_bn.npz',
                targets,[c['name'] for c in adapter.CONDITIONS],mapping['global_class_ids'])
    # Match the saved raw validation accuracy, then compare with existing calibrated predictions.
    for audit in checkpoint_audit:
        actual=results[audit['method']]['raw_bn']['seen']['clean']
        require(abs(actual-audit['saved_raw_val_accuracy_pct'])<1e-6,
                f"Raw validation replay mismatch for {audit['method']}")
    table=endpoint_rows(results)
    controls.write_csv(out/'endpoints.csv',table)
    deltas=[]
    for method in METHODS:
        for suite in ('seen','unseen'):
            for condition,value in results[method]['raw_bn'][suite].items():
                cal=results[method]['calibrated_bn'][suite][condition]
                deltas.append({'method':method,'suite':suite,'condition':condition,'raw_pct':value,
                               'calibrated_pct':cal,'cal_minus_raw_pp':cal-value})
    controls.write_csv(out/'bn_deltas.csv',deltas)
    shutil.copyfile(Path(__file__),out/Path(__file__).name)
    shutil.copyfile(run/'val_sample_index.csv',out/'val_sample_index.csv')
    original.write_json(out/'result.json',{'completed':True,'run_dir':str(run),'results':results,
        'checkpoint_audit':checkpoint_audit,'source_sha256':sha(Path(__file__)),
        'test_evaluated':False,'confusion_evaluated':False,'training_steps':0})
    logging.info('DONE: %s',out)
    print((out/'endpoints.csv').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':
    main()
