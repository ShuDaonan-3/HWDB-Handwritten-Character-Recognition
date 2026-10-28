"""FP32 pilot: normalized softmax with degraded-view Top1/Top3 cosine hinge. No test evaluation."""
import argparse
import csv
import gc
import importlib.util
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

ROOT = Path("/workspace/shouxieti")
def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(value, message):
    if not value:
        raise RuntimeError(message)


def fixed_margin_loss(scores, labels, selected=3):
    # Match the teacher-margin hard-negative selection exactly, including its pool.
    with torch.no_grad():
        ranking = scores.detach().clone()
        ranking.scatter_(1, labels[:, None], float("-inf"))
        pool = ranking.topk(min(10, scores.shape[1] - 1), dim=1).indices
        positions = scores.detach().gather(1, pool).topk(min(selected, pool.shape[1]), dim=1).indices
        rivals = pool.gather(1, positions)
    gap = scores.gather(1, labels[:, None]) - scores.gather(1, rivals)
    violation = F.relu(0.2 - gap)
    return violation.mean(), (violation.detach() > 0).any(1).float().mean()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def require(value, message):
    if not value:
        raise RuntimeError(message)

def fixed_margin_loss(scores, labels, selected=3):
    # Match the teacher-margin hard-negative selection exactly, including its pool.
    with torch.no_grad():
        ranking = scores.detach().clone()
        ranking.scatter_(1, labels[:, None], float("-inf"))
        pool = ranking.topk(min(10, scores.shape[1] - 1), dim=1).indices
        positions = scores.detach().gather(1, pool).topk(min(selected, pool.shape[1]), dim=1).indices
        rivals = pool.gather(1, positions)
    gap = scores.gather(1, labels[:, None]) - scores.gather(1, rivals)
    violation = F.relu(0.2 - gap)
    return violation.mean(), (violation.detach() > 0).any(1).float().mean()

def features_of(model, images):
    x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
    return torch.flatten(model.avgpool(x), 1)

class InferenceHead(torch.nn.Module):
    def __init__(self, backbone, normalized):
        super().__init__()
        self.backbone = backbone
        self.normalized = normalized

    def forward(self, images):
        features = features_of(self.backbone, images)
        if self.normalized:
            return 30.0 * (F.normalize(features.float(), dim=1) @
                           F.normalize(self.backbone.fc.weight.float(), dim=1).T)
        return self.backbone.fc(features)

def margin_logits(scores, labels, margin):
    return 30.0 * (scores - F.one_hot(labels, scores.shape[1]).to(scores.dtype) * margin)

def train_one(method, parent, args, rows, data_config, shared, recipe, initial_hash, reference,
                helper, controls, robust, transfer, candidate, repeats):
    require(args.no_amp, "Combination pilot requires FP32")
    out = parent / method
    out.mkdir()
    controls.seed_everything(args.seed, args.threads)
    model = helper.make_model()
    require(controls.state_fingerprint(model) == initial_hash, "Fixed-margin initialization differs from baseline")
    spec = METHODS[method]
    normalized = spec["head"] == "normalized"
    if normalized:
        model.fc.bias.requires_grad_(False)
    prediction_model = InferenceHead(model, normalized)
    loader = controls.make_loader(helper, rows["train"], 128, args.workers, True, args.seed)
    val = controls.make_loader(helper, rows["val"], 128, args.workers, False, args.seed + 1)
    calibration = shared["bn_calibration"]
    cal_loader = controls.make_loader(helper, [rows["train"][i] for i in calibration["indices"]],
                                      128, args.workers, False, args.seed + 2000)
    generator = torch.Generator().manual_seed(args.seed + 3000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    config = {**recipe, "method": method, "inference_head": spec["head"], "inference_scale": 30 if normalized else None, "training_spec": spec, "initial_state_sha256": initial_hash,
              "teacher_checkpoint": None, "teacher_checkpoint_sha256": None}
    helper.write_json(out / "config.json", config)
    history = []
    counts_total = np.zeros(len(robust.CONDITIONS), dtype=np.int64)
    logging.info("START %s seed=%d; 10 epochs; spec=%s", method, args.seed, spec)
    for epoch in range(1, 11):
        started = time.perf_counter()
        model.train()
        n_total = skipped = clean_correct = corrupt_correct = 0
        ce_sum = aux_sum = active_sum = 0.0
        coefficient = 0.0 if method == "norm_softmax" else candidate.auxiliary_weight(epoch)
        lr = optimizer.param_groups[0]["lr"]
        for step, (images, labels) in enumerate(loader, 1):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            second, counts = controls.second_view(images, generator, robust)
            counts_total += counts
            n = len(labels)
            optimizer.zero_grad(set_to_none=True)
            # Explicit FP32 backbone and normalized head; affine logits are unused.
            unused_affine, features = candidate.forward_features(model, torch.cat((images, second)))
            scores = candidate.cosine_scores(model, features)
            logits = 30.0 * scores
            ce = F.cross_entropy(logits, torch.cat((labels, labels)))
            if spec["selected"]:
                auxiliary, active = fixed_margin_loss(scores[n:], labels, spec["selected"])
            else:
                auxiliary = active = ce.detach().new_zeros(())
            loss = ce if coefficient == 0 else ce + coefficient * auxiliary
            require(bool(torch.isfinite(loss)) and bool(torch.isfinite(auxiliary)), "Non-finite loss")
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            n_total += n
            ce_sum += ce.item() * n
            aux_sum += auxiliary.item() * n
            active_sum += active.item() * n
            clean_correct += (logits[:n].argmax(1) == labels).sum().item()
            corrupt_correct += (logits[n:].argmax(1) == labels).sum().item()
            if step == 1 or step % 200 == 0:
                logging.info("%s seed=%d epoch=%d/10 step=%d/%d CE=%.4f aux=%.4f weight=%.2f",
                             method, args.seed, epoch, step, len(loader), ce_sum / n_total, aux_sum / n_total, coefficient)
        train_seconds = time.perf_counter() - started
        raw_loss, raw_acc = helper.evaluate(prediction_model, val, torch.nn.CrossEntropyLoss(), False)
        calibrated = controls.calibrate_copy(prediction_model, cal_loader)
        cal_loss, cal_acc = helper.evaluate(calibrated, val, torch.nn.CrossEntropyLoss(), False)
        history.append({"epoch": epoch, "lr": lr, "auxiliary_weight": coefficient,
                        "train_ce": ce_sum / n_total, "train_aux": aux_sum / n_total,
                        "train_active": active_sum / n_total, "train_clean_accuracy": clean_correct / n_total,
                        "train_corrupt_accuracy": corrupt_correct / n_total,
                        "val_raw_loss": raw_loss, "val_raw_accuracy": raw_acc,
                        "val_calibrated_loss": cal_loss, "val_calibrated_accuracy": cal_acc,
                        "train_source_images": n_total, "model_input_images": 2 * n_total,
                        "teacher_input_images": 0, "attempted_steps": len(loader), "skipped_amp_steps": skipped,
                        "train_seconds": train_seconds, "total_seconds": time.perf_counter() - started})
        controls.write_csv(out / "metrics.csv", history)
        logging.info("EPOCH %s seed=%d %d: raw=%.2f%% calibrated=%.2f%% seconds=%.1f skipped=%d",
                     method, args.seed, epoch, 100 * raw_acc, 100 * cal_acc, history[-1]["total_seconds"], skipped)
        if epoch == 1 and method != "norm_softmax":
            matches = abs(raw_acc - reference["raw"]) < 1e-6 and abs(cal_acc - reference["calibrated"]) < 1e-6
            helper.write_json(out / "warmup_check.json", {"matches_norm_softmax_epoch1": matches,
                              "raw_accuracy": raw_acc, "calibrated_accuracy": cal_acc, "reference": reference})
            require(matches, "CE-only epoch 1 did not reproduce the same-seed normalized-softmax baseline")
        scheduler.step()
        helper.save_checkpoint(out / "last.pt", {"model": model.state_dict(), "epoch": epoch,
                               "config": config, "val_accuracy": raw_acc})
        if epoch != 10:
            del calibrated
    helper.save_checkpoint(out / "epoch10_bn_calibrated.pt", {"model": calibrated.backbone.state_dict(), "epoch": 10,
                           "config": config, "val_accuracy": cal_acc, "bn_calibration": calibration})
    metrics = final_evaluation(prediction_model, calibrated, out, rows, args, data_config, helper, controls, robust, transfer)
    require(abs(metrics["raw_bn"]["seen"]["clean"] - 100 * raw_acc) < 1e-8, "Raw validation replay mismatch")
    helper.write_json(out / "result.json", {"completed": True, "method": method, "inference_head": spec["head"], "inference_scale": 30 if normalized else None, "training_spec": spec, "metrics": metrics,
                      "test_evaluated": False, "confusion_evaluated": False, "teacher_input_images": 0,
                      "student_input_images": sum(r["model_input_images"] for r in history),
                      "skipped_amp_steps": sum(r["skipped_amp_steps"] for r in history),
                      "second_view_counts": {c["name"]: int(counts_total[i]) for i, c in enumerate(robust.CONDITIONS)}})
    return metrics

# GENERATED TRAINING FUNCTIONS ARE INSERTED ABOVE THIS LINE

METHODS = {
    'norm_softmax': {'head':'normalized','margin':0.0,'selected':0},
    'norm_hinge_top1': {'head':'normalized','margin':0.0,'selected':1},
    'norm_hinge_top3': {'head':'normalized','margin':0.0,'selected':3},
}
PROTOCOL = {
    'version':1,'classes':500,'seed':42,'epochs':10,'precision':'FP32',
    'methods':METHODS,'scale':30.0,'hinge_margin':0.2,'negative_pool':10,
    'classification':'normalized bias-free cosine CE over paired clean and degraded views, no target margin',
    'auxiliary':'mean relu(0.2-(cos_y-cos_j)) over hardest Top1/Top3 incorrect classes on degraded views only',
    'auxiliary_weight':'epoch1=0; epoch2=0.5; epochs3-10=1',
    'checkpoint':'fixed epoch10; no best-epoch selection',
    'primary_bn_variant':'raw_bn for ALL methods; calibrated_bn is diagnostic only',
    'primary':'mean accuracy over frozen 8 training-unseen synthetic validation conditions',
    'secondary':'clean, individual conditions, six seen-condition mean',
    'selection':'prespecified three arms; report all outcomes, including regressions',
    'interpretation':'exploratory single-seed pilot; new data performance and novelty not established',
    'test_evaluated':False,'confusion_evaluated':False,
}


def final_evaluation(raw,calibrated,directory,rows,args,data_config,helper,controls,robust,transfer):
    loader=controls.make_loader(helper,rows['val'],128,args.workers,False,args.seed+1)
    targets=np.asarray([r[4] for r in rows['val']],dtype=np.int64)
    result={'raw_bn':{},'calibrated_bn':{}}
    for suite,adapter in (('seen',robust),('unseen',transfer.ProtocolAdapter(robust.corrupt))):
        out=directory/suite
        out.mkdir()
        values=controls.evaluate_conditions({'raw_bn':raw,'calibrated_bn':calibrated},loader,targets,adapter,out,data_config)
        for variant in result:
            result[variant][suite]={r['condition']:r['accuracy_pct'] for r in values if r['bn_variant']==variant}
    return result


def write_tables(parent,results,controls):
    endpoints,detail=[],[]
    for method,variants in results.items():
        for variant,suites in variants.items():
            require(abs(suites['seen']['clean']-suites['unseen']['clean'])<1e-9,'Clean suite mismatch')
            row={'method':method,'bn_variant':variant,'clean_pct':suites['seen']['clean'],
                'mean_of_6_seen_pct':float(np.mean([v for k,v in suites['seen'].items() if k!='clean'])),
                'mean_of_8_unseen_pct':float(np.mean([v for k,v in suites['unseen'].items() if k!='clean'])),
                'lowres_20_pct':suites['unseen']['lowres_20'],'blur_s2.0_pct':suites['unseen']['blur_s2.0']}
            endpoints.append(row)
            for suite,values in suites.items():
                for condition,value in values.items():
                    baseline=results['norm_softmax'][variant][suite][condition]
                    detail.append({'method':method,'bn_variant':variant,'suite':suite,'condition':condition,
                                   'accuracy_pct':value,'delta_vs_norm_softmax_pp':value-baseline})
    controls.write_csv(parent/'endpoints.csv',endpoints)
    controls.write_csv(parent/'comparison.csv',detail)


def self_checks(helper,ablation,repeats):
    ablation.self_checks(helper,repeats)
    labels=torch.tensor([0],device='cuda')
    scores=torch.tensor([[.1,.2,0.,-.2]],device='cuda',requires_grad=True)
    for k in (1,3):
        loss,_=fixed_margin_loss(scores,labels,k)
        reference,_=ablation.fixed_margin_loss(scores,labels,k)
        torch.testing.assert_close(loss,reference,atol=0,rtol=0)
        grad=torch.autograd.grad(loss,scores,retain_graph=True)[0]
        refgrad=torch.autograd.grad(reference,scores,retain_graph=True)[0]
        torch.testing.assert_close(grad,refgrad,atol=0,rtol=0)
    # A clean row must not affect the degraded-only auxiliary.
    clean=scores.detach().clone()+.5
    paired=torch.cat((clean,scores),0)
    v,_=fixed_margin_loss(paired[1:],labels,3)
    torch.testing.assert_close(v,repeats.fixed_margin_loss(scores,labels)[0])
    logging.info('COMBINATION SELF CHECKS PASS: normalized inference, Top1/Top3 loss/gradient, degraded-only auxiliary')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablations-run',type=Path,required=True)
    opts=parser.parse_args()
    old=opts.ablations_run.resolve()
    require(old.is_relative_to(ROOT.resolve()),'Reference outside project')
    shared=read_json(old/'config.json')
    require(read_json(old/'result.json')['completed'],'Pilot ablation run incomplete')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('0','1') and torch.cuda.is_available()
            and torch.cuda.device_count()==1,'Expose one authorized GPU')
    require(str(torch.__version__)==shared['torch'] and str(torchvision.__version__)==shared['torchvision'],
            'Environment changed')
    import hashlib
    sources={name:(old/Path(item['path']).name,item['sha256']) for name,item in shared['source_snapshots'].items()}
    sources['ablation']=(old/'train_pilot_margin_ablations.py',shared['source_sha256'])
    for name,(path,expected) in sources.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest()==expected,f'Changed snapshot: {name}')
    modules={n:load_module('normhinge_'+n,p) for n,(p,_) in sources.items()}
    helper,controls,robust,transfer,candidate,repeats,ablation=[modules[n] for n in
        ('helper','controls','robust','transfer','candidate','repeats','ablation')]
    rows,data_config,hashes=helper.load_data(ROOT)
    require(hashes==shared['data_sha256'],'Data/splits changed')
    require(len(rows['train'])==99594 and len(rows['val'])==9947 and
            len(data_config['pilot_classes.json']['global_class_ids'])==500,'Expected original pilot500')
    for filename,expected in (('bn_calibration_pilot_v1.json',shared['bn_calibration']),
                              ('pilot_robustness_v1.json',shared['seen_protocol']),
                              ('pilot_transfer_v1.json',shared['unseen_protocol'])):
        require(read_json(ROOT/'configs'/filename)==expected,f'Changed protocol: {filename}')
    args=argparse.Namespace(**shared['args'])
    require(args.seed==42 and args.epochs==10 and args.batch_size==128,'Unexpected original recipe')
    args.no_amp=True
    parent=ROOT/'outputs'/f'pilot_normalized_hinge_{datetime.now():%Y%m%d_%H%M%S_%f}_seed42'
    parent.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),
        logging.FileHandler(ROOT/'logs'/f'{parent.name}.log',encoding='utf-8')])
    logging.info('Output: %s; 3 FP32 jobs, 500 classes, %d train / %d val; GPU=%s',parent,len(rows['train']),len(rows['val']),torch.cuda.get_device_name(0))
    controls.seed_everything(args.seed,args.threads)
    self_checks(helper,ablation,repeats)
    robust.fixed_json(ROOT/'configs/pilot_normalized_hinge_v1.json',PROTOCOL)
    recipe={**shared,'args':vars(args),'protocol':PROTOCOL,'ablations_run':str(old),
        'source_sha256':helper.sha256(Path(__file__)),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'cudnn_version':torch.backends.cudnn.version(),'cudnn_enabled':torch.backends.cudnn.enabled,
        'source_snapshots':{n:{'path':str(p),'sha256':h} for n,(p,h) in sources.items()}}
    helper.write_json(parent/'config.json',recipe)
    shutil.copyfile(Path(__file__),parent/Path(__file__).name)
    for path,_ in sources.values():
        shutil.copyfile(path,parent/path.name)
    shutil.copyfile(old/'val_sample_index.csv',parent/'val_sample_index.csv')
    control_run=Path(shared['controls_run']).resolve()
    require(control_run.is_relative_to(ROOT.resolve()),'Original controls outside project')
    initial_hash=read_json(control_run/'initial_state.json')['sha256']
    reference=None
    results={}
    for method in METHODS:
        results[method]=train_one(method,parent,args,rows,data_config,shared,recipe,initial_hash,reference,
            helper,controls,robust,transfer,candidate,repeats)
        if method=='norm_softmax':
            reference=repeats.warmup_reference(parent/method)
        write_tables(parent,results,controls)
        gc.collect()
        torch.cuda.empty_cache()
    helper.write_json(parent/'result.json',{'completed':True,'results':results,'classes':500,'seed':42,
        'precision':'FP32','primary_bn_variant':'raw_bn','new_training_jobs':3,
        'test_evaluated':False,'confusion_evaluated':False})
    logging.info('DONE: %s',parent)
    print((parent/'endpoints.csv').read_text(encoding='utf-8'),flush=True)


if __name__=='__main__':
    main()
