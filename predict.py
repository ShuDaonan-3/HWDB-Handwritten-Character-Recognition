import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from project import ROOT,modules,load_checkpoint,read_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    m=modules()
    model=load_checkpoint(args.checkpoint,args.device,m)
    image=m['helper'].center_image(Image.open(args.image).convert('L'))
    x=torch.from_numpy(np.array(image,dtype=np.uint8,copy=True)).float()[None,None]/127.5-1
    with torch.inference_mode():
        scores,ids=model(x.to(args.device)).softmax(1).topk(5,dim=1)
    classes=read_json(ROOT/'configs/classes.json')['classes']
    print(json.dumps([{'character':classes[i],'class_id':i,'softmax_score':s}
        for i,s in zip(ids[0].tolist(),scores[0].tolist())],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
