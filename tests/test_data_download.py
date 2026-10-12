import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('download_data', Path(__file__).parents[1] / 'tools/download_data.py')
download_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_data)


def bundle(tmp_path, member='expression/values.txt'):
    path = tmp_path / 'input.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(member, 'example measurements')
    entry = {'filename': path.name, 'directory': 'expression', 'bytes': path.stat().st_size,
             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return path, entry


def test_validated_bundle_is_extracted(tmp_path):
    archive, entry = bundle(tmp_path)
    download_data.extract_verified(archive, entry, tmp_path / 'output')
    assert (tmp_path / 'output/expression/values.txt').read_text() == 'example measurements'


def test_corruption_is_rejected_before_extraction(tmp_path):
    archive, entry = bundle(tmp_path)
    archive.write_bytes(archive.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Checksum or size'):
        download_data.extract_verified(archive, entry, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('member', ['../outside.txt', '/absolute.txt', 'expression/../../outside.txt', 'other/data.txt'])
def test_unexpected_archive_paths_are_rejected(tmp_path, member):
    archive, entry = bundle(tmp_path, member)
    with pytest.raises(ValueError, match='archive member'):
        download_data.extract_verified(archive, entry, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()
