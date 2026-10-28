"""Export source/results plus the two main full3755 raw-BN checkpoints. No images."""
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

ROOT=Path('/workspace/shouxieti')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    candidates=[]
    for folder in sorted((ROOT/'outputs').glob('full3755_normalized_hinge_*')):
        result=folder/'result.json'
        if result.is_file() and json.loads(result.read_text()) .get('completed'):
            candidates.append(folder)
    if not candidates:
        raise RuntimeError('No completed full3755_normalized_hinge run found')
    proposed=candidates[-1]
    config=json.loads((proposed/'config.json').read_text())
    baseline=Path(config['full_run']).resolve()
    if not baseline.is_relative_to(ROOT.resolve()):
        raise RuntimeError('Baseline path outside project')
    models={
        'baseline_normalized_softmax':baseline/'norm_softmax/last.pt',
        'proposed_normalized_hinge_top1':proposed/'norm_hinge_top1/last.pt',
    }
    for name,path in models.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata=json.loads((path.parent/'config.json').read_text())
        if metadata['args']['seed']!=42 or not metadata['args']['no_amp']:
            raise RuntimeError(f'Unexpected training recipe: {name}')
    out=ROOT/'exports'
    out.mkdir(exist_ok=True)
    archive=out/f'course_github_{datetime.now():%Y%m%d_%H%M%S}.zip'
    manifest={'classes':3755,'seed':42,'epochs':10,'precision':'FP32','bn_variant':'raw_bn',
        'checkpoint_selection':'fixed epoch10; last.pt','proposed_run':str(proposed),
        'baseline_run':str(baseline),'models':{},
        'scope':'Main 3755-class baseline/proposed weights. Pilot/ablation weight files are not included.',
        'data':'No original GNT, image files, paper PDFs, or dataset archives included.'}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for name,path in models.items():
            destination=f'weights/{name}.pt'
            manifest['models'][name]={'source':str(path),'archive_path':destination,
                'bytes':path.stat().st_size,'sha256':digest(path),
                'inference':'normalized cosine head, scale30, no bias, no ground-truth margin'}
            z.write(path,destination)
            print('Included checkpoint:',name,flush=True)
        # Whitelist code and experiment metadata; never recurse into data/raw or caches.
        for directory in (ROOT/'src',ROOT/'configs',ROOT/'outputs'):
            if not directory.exists():
                continue
            for path in sorted(directory.rglob('*')):
                if not path.is_file() or path.suffix.lower() not in ('.py','.json','.csv'):
                    continue
                if '__pycache__' in path.parts:
                    continue
                resolved=path.resolve()
                if not resolved.is_relative_to(ROOT.resolve()):
                    raise RuntimeError(f'Unexpected path outside project: {path}')
                z.write(path,path.relative_to(ROOT).as_posix())
        z.writestr('EXPORT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:
            raise RuntimeError(f'Archive verification failed: {bad}')
        for item in manifest['models'].values():
            if z.getinfo(item['archive_path']).file_size!=item['bytes']:
                raise RuntimeError('Checkpoint size mismatch')
    checksum=digest(archive)
    archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n',encoding='utf-8')
    print('DONE:',archive,flush=True)
    print('Size MiB:',round(archive.stat().st_size/1024**2,1),flush=True)
    print('SHA256:',checksum,flush=True)
    print('Upload this ZIP for repository preparation. Main weights are raw-BN epoch10, not calibrated weights.')


if __name__=='__main__':
    main()
