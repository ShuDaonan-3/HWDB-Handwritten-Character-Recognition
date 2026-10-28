"""Download the two published checkpoint ZIPs, verify SHA256, then extract."""
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE='https://raw.githubusercontent.com/ShuDaonan-3/HWDB-Handwritten-Character-Recognition/main/'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest=json.loads((ROOT/'weights/manifest.json').read_text(encoding='utf-8'))
    for name,item in manifest['models'].items():
        target=ROOT/item['archive_path']
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file() and sha(target)==item['sha256']:
            print('Verified:',target.name)
            continue
        archive=ROOT/item['download_zip']
        if not archive.is_file() or sha(archive)!=item['zip_sha256']:
            temp=archive.with_suffix('.download')
            if item.get('download_parts'):
                with temp.open('wb') as combined:
                    for part in item['download_parts']:
                        path=ROOT/part['path']
                        if not path.is_file() or sha(path)!=part['sha256']:
                            partial=path.with_suffix('.download')
                            with urllib.request.urlopen(BASE+part['path'],timeout=120) as response,partial.open('wb') as f:
                                shutil.copyfileobj(response,f)
                            if sha(partial)!=part['sha256']:
                                raise RuntimeError('Checkpoint part checksum mismatch: '+part['path'])
                            partial.replace(path)
                        with path.open('rb') as source:
                            shutil.copyfileobj(source,combined)
            else:
                with urllib.request.urlopen(BASE+item['download_zip'],timeout=120) as response,temp.open('wb') as f:
                    shutil.copyfileobj(response,f)
            if sha(temp)!=item['zip_sha256']:
                raise RuntimeError('Downloaded archive checksum mismatch: '+name)
            temp.replace(archive)
        with zipfile.ZipFile(archive) as z:
            if z.namelist()!=[target.name] or z.getinfo(target.name).file_size!=item['bytes']:
                raise RuntimeError('Unexpected checkpoint ZIP contents: '+name)
            temp=target.with_suffix('.extracting')
            with z.open(target.name) as source,temp.open('wb') as dest:
                shutil.copyfileobj(source,dest)
            if sha(temp)!=item['sha256']:
                raise RuntimeError('Extracted checkpoint checksum mismatch: '+name)
            temp.replace(target)
        print('Verified:',target.name)


if __name__=='__main__':
    main()
