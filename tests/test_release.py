import ast
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_entry_point_syntax(self):
        for path in ROOT.glob('*.py'):
            compile(ast.parse(path.read_text(encoding='utf-8')),str(path),'exec')

    def test_class_order(self):
        expected=[bytes((a,b)).decode('gb2312') for a in range(0xB0,0xD8)
                  for b in range(0xA1,0xFF) if a<0xD7 or b<=0xF9]
        actual=json.loads((ROOT/'configs/classes.json').read_text(encoding='utf-8'))['classes']
        self.assertEqual(len(expected),3755)
        self.assertEqual(actual,expected)

    def test_checkpoint_part_hashes(self):
        manifest=json.loads((ROOT/'weights/manifest.json').read_text(encoding='utf-8'))
        for item in manifest['models'].values():
            combined=hashlib.sha256()
            for part in item['download_parts']:
                data=(ROOT/part['path']).read_bytes()
                self.assertEqual(len(data),part['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(),part['sha256'])
                combined.update(data)
            self.assertEqual(combined.hexdigest(),item['zip_sha256'])

    def test_offline_weight_restore(self):
        spec=importlib.util.spec_from_file_location('restore_test',ROOT/'download_weights.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root=Path(folder); (root/'weights').mkdir(); module.ROOT=root
            payload=b'checkpoint fixture'
            archive=root/'weights/tiny.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('tiny.pt',payload)
            data=archive.read_bytes()
            parts=[]
            for i,part in enumerate((data[:10],data[10:])):
                name=f'weights/tiny.zip.part{i:02d}'
                (root/name).write_bytes(part)
                parts.append({'path':name,'sha256':hashlib.sha256(part).hexdigest()})
            archive.unlink()
            item={'archive_path':'weights/tiny.pt','download_zip':'weights/tiny.zip',
                  'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),
                  'zip_sha256':hashlib.sha256(data).hexdigest(),'download_parts':parts}
            (root/'weights/manifest.json').write_text(json.dumps({'models':{'tiny':item}}))
            module.main()
            self.assertEqual((root/'weights/tiny.pt').read_bytes(),payload)


if __name__=='__main__':
    unittest.main()
