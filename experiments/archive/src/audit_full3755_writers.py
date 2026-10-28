"""Descriptive paired writer audit of existing raw-BN validation predictions.

One training seed, 20 held-out validation writers. Repeated image conditions
are not independent samples. Leave-one-writer-out ranges are NOT confidence
intervals. No p values or test-set measurements are produced.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/shouxieti')
METHODS = ('aug_ce','margin_fixed','norm_softmax','cosface_m010')
PAIRS = (('aug_ce','margin_fixed'),('norm_softmax','margin_fixed'),
         ('aug_ce','norm_softmax'),('norm_softmax','cosface_m010'))
ENDPOINTS = (('clean','seen',['clean']),('mean_of_6_seen','seen',None),
             ('mean_of_8_unseen','unseen',None),('lowres_20','unseen',['lowres_20']),
             ('blur_s2.0','unseen',['blur_s2.0']))


def require(value,message):
    if not value:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path,rows):
    require(bool(rows),'Empty output table')
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_endpoint(a,b,writers):
    """a/b: correctness matrices [conditions, SAME images]. Positive means b wins."""
    require(a.shape==b.shape and a.ndim==2 and a.shape[1]==len(writers),'Pair shape mismatch')
    unique=np.unique(writers)
    require(len(unique)>1,'Need multiple writers')
    gains=(~a)&b
    losses=a&(~b)
    per_writer=[]
    loo=[]
    for writer in unique:
        mask=writers==writer
        g,l=int(gains[:,mask].sum()),int(losses[:,mask].sum())
        delta=100*(g-l)/(a.shape[0]*int(mask.sum()))
        per_writer.append({'writer_id':int(writer),'images':int(mask.sum()),
            'image_condition_evaluations':int(a.shape[0]*mask.sum()),
            'a_accuracy_pct':float(a[:,mask].mean()*100),'b_accuracy_pct':float(b[:,mask].mean()*100),
            'b_minus_a_pp':delta,'gains':g,'losses':l,'net_correct':g-l})
        keep=~mask
        loo.append(100*int(gains[:,keep].sum()-losses[:,keep].sum())/int(a[:,keep].size))
    deltas=[r['b_minus_a_pp'] for r in per_writer]
    total_gain,total_loss=int(gains.sum()),int(losses.sum())
    summary={'images':len(writers),'conditions':a.shape[0],'writers':len(unique),
        'image_condition_evaluations':int(a.size),
        'a_accuracy_pct':float(a.mean()*100),'b_accuracy_pct':float(b.mean()*100),
        'b_minus_a_pp':100*(total_gain-total_loss)/a.size,
        'equal_writer_mean_delta_pp':float(np.mean(deltas)),
        'gains':total_gain,'losses':total_loss,'net_correct':total_gain-total_loss,
        'writers_better':sum(d>0 for d in deltas),'writers_tied':sum(d==0 for d in deltas),
        'writers_worse':sum(d<0 for d in deltas),
        'leave_one_writer_out_min_pp':float(min(loo)),
        'leave_one_writer_out_max_pp':float(max(loo)),
        'largest_absolute_writer_net':max(abs(r['net_correct']) for r in per_writer)}
    return summary,per_writer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-run',required=True,type=Path)
    opts=parser.parse_args()
    run=opts.audit_run.resolve()
    require(run.is_relative_to(ROOT.resolve()),'Audit run outside project')
    prior=read_json(run/'result.json')
    require(prior['completed'] and prior['training_steps']==0,'Incomplete BN audit')
    original=Path(prior['run_dir']).resolve()
    require(original.is_relative_to(ROOT.resolve()),'Training run outside project')
    config=read_json(original/'config.json')
    require(config['args']['seed']==42 and config['args']['no_amp'] is True,'Expected seed42 FP32 comparison')
    index=run/'val_sample_index.csv'
    require(sha(index)==sha(original/'val_sample_index.csv'),'Validation index changed')
    with index.open(encoding='utf-8',newline='') as f:
        samples=list(csv.DictReader(f))
    require(len(samples)==74760,'Unexpected validation size')
    require([int(r['row']) for r in samples]==list(range(len(samples))),'Index row order mismatch')
    targets=np.asarray([int(r['global_class_id']) for r in samples],dtype=np.int64)
    writers=np.asarray([int(r['writer_id']) for r in samples],dtype=np.int64)
    require(len(np.unique(writers))==20,'Expected 20 validation writers')
    require(all(int(r['local_class_id'])==int(r['global_class_id']) for r in samples),'Unexpected local mapping')
    mapping=config['evaluation_class_mapping']
    require(mapping['global_class_ids']==list(range(3755)),'Unexpected full class mapping')
    require(all(r['character']==mapping['classes'][int(r['global_class_id'])] for r in samples),'Index character mismatch')
    correct,names,provenance={}, {}, []
    for suite,expected in (('seen',config['seen_protocol']),('unseen',config['unseen_protocol'])):
        names[suite]=[c['name'] for c in expected['conditions']]
        require(names[suite][0]=='clean','Missing first clean condition')
        correct[suite]={}
        for method in METHODS:
            path=run/suite/f'predictions_{method}.npz'
            with np.load(path,allow_pickle=False) as data:
                require(np.array_equal(data['targets'],targets),'Prediction target/order mismatch')
                require(data['conditions'].tolist()==names[suite],'Condition mismatch')
                require(np.array_equal(data['global_class_ids'],mapping['global_class_ids'])
                        and data['classes'].tolist()==mapping['classes'],'Class mapping mismatch')
                predictions=data['predictions']
                require(predictions.shape==(len(names[suite]),len(samples)),'Prediction shape mismatch')
                require(np.issubdtype(predictions.dtype,np.integer) and np.all((predictions>=0)&(predictions<3755)),
                        'Invalid predicted class IDs')
                correct[suite][method]=predictions==targets[None,:]
                for i,name in enumerate(names[suite]):
                    accuracy=float(correct[suite][method][i].mean()*100)
                    require(abs(accuracy-prior['results'][method]['raw_bn'][suite][name])<1e-8,
                            f'Previous accuracy not reproduced: {method}/{suite}/{name}')
            provenance.append({'method':method,'suite':suite,'path':str(path),'sha256':sha(path)})
    for method in METHODS:
        require(np.array_equal(correct['seen'][method][0],correct['unseen'][method][0]),'Clean correctness differs')
    summaries,details,individual=[], [], []
    for a,b in PAIRS:
        for endpoint,suite,selected in ENDPOINTS:
            condition_names=selected if selected is not None else [n for n in names[suite] if n!='clean']
            indices=[names[suite].index(n) for n in condition_names]
            summary,per_writer=paired_endpoint(correct[suite][a][indices],correct[suite][b][indices],writers)
            labels={'endpoint':endpoint,'a':a,'b':b}
            summaries.append({**labels,**summary})
            details.extend({**labels,**row} for row in per_writer)
        for suite in ('seen','unseen'):
            for i,condition in enumerate(names[suite]):
                summary,_=paired_endpoint(correct[suite][a][i:i+1],correct[suite][b][i:i+1],writers)
                individual.append({'suite':suite,'condition':condition,'a':a,'b':b,**summary})
    out=ROOT/'outputs'/f'full3755_writer_audit_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True)
    write_csv(out/'summary.csv',summaries)
    write_csv(out/'per_writer.csv',details)
    write_csv(out/'per_condition.csv',individual)
    (out/'result.json').write_text(json.dumps({'completed':True,'source_audit':str(run),'source_sha256':sha(Path(__file__)),
        'val_index_sha256':sha(index),'prediction_provenance':provenance,'seed':42,'bn_variant':'raw_bn',
        'interpretation':'Descriptive single-seed validation audit. Repeated conditions are not independent images. '
                         'Leave-one-writer-out ranges are sensitivity summaries, NOT confidence intervals. '
                         'No claim of significance, novelty, or performance on new data.',
        'training_steps':0,'test_evaluated':False,'confusion_evaluated':False},indent=2),encoding='utf-8')
    print('PASS: all raw accuracies reproduced; 74760 images / 20 writers; paired sample/class/condition alignment.')
    for r in summaries:
        print(f"{r['endpoint']:18s} {r['b']:14s} vs {r['a']:14s} "
              f"delta={r['b_minus_a_pp']:+.4f}pp gains/losses={r['gains']}/{r['losses']} "
              f"writers(+/0/-)={r['writers_better']}/{r['writers_tied']}/{r['writers_worse']} "
              f"leave-one-writer-out=[{r['leave_one_writer_out_min_pp']:+.4f},{r['leave_one_writer_out_max_pp']:+.4f}]pp")
    print('Counts for condition means are image-condition evaluations, not independent images.')
    print('Leave-one-writer-out ranges are sensitivity summaries, NOT confidence intervals.')
    print('DONE:',out)


if __name__=='__main__':
    main()
