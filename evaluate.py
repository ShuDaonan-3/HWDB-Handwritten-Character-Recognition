"""Evaluate a released raw-BN checkpoint on the fixed validation split."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from project import ROOT, modules, load_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--suite',choices=['clean','seen','unseen'],default='unseen')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output',type=Path,default=Path('outputs/evaluation.csv'))
    opts=parser.parse_args()
    m=modules()
    m['controls'].seed_everything(42,4)
    rows,_,_,_=m['full'].load_full_data(ROOT,m['helper'])
    dataset=m['helper'].GNTDataset(rows['val'])
    loader=DataLoader(dataset,batch_size=128,shuffle=False,num_workers=0)
    model=load_checkpoint(opts.checkpoint,opts.device,m)
    adapter=m['transfer'].ProtocolAdapter(m['robust'].corrupt)
    conditions=adapter.CONDITIONS if opts.suite=='unseen' else m['robust'].CONDITIONS
    if opts.suite=='clean':
        conditions=conditions[:1]
    results=[]
    with torch.inference_mode():
        for condition in conditions:
            correct=total=0
            for images,labels in loader:
                images=adapter.corrupt(images.to(opts.device),condition)
                logits=model(images)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Non-finite evaluation logits')
                correct+=(logits.argmax(1).cpu()==labels).sum().item()
                total+=len(labels)
            row={'condition':condition['name'],'samples':total,'accuracy_pct':100*correct/total,'errors':total-correct}
            results.append(row)
            print(json.dumps(row),flush=True)
    dataset.close()
    opts.output.parent.mkdir(parents=True,exist_ok=True)
    with opts.output.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)


if __name__=='__main__':
    main()
