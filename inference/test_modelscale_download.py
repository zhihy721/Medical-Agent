import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from inference.modelscope_download_worker import validate_manifest, file_sha


class DownloadIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.files = [dict(Path=p, Type='blob', Revision='a'*40, Sha256='b'*64, Size=1)
                      for p in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                                'model.safetensors.index.json', 'LICENSE', 'model.safetensors']]

    def test_rejects_path_traversal_and_duplicate_files(self):
        for path in ['../other-model/config.json', '/home/other/config.json', 'config.json']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_manifest(self.files + [{**self.files[0], 'Path': path}])

    def test_requires_official_hash_revision_and_weight_inventory(self):
        for key, value in [('Sha256', ''), ('Revision', ''), ('Size', -1)]:
            files = copy.deepcopy(self.files)
            files[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(files)
        with self.assertRaises(ValueError):
            validate_manifest(self.files[:-1])

    def test_streaming_digest(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'file'
            p.write_bytes(b'known model bytes')
            self.assertEqual(file_sha(p), hashlib.sha256(b'known model bytes').hexdigest())


if __name__=='__main__': unittest.main()
