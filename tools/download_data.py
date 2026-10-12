"""Download and verify the three prepared CellCanon input bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

from huggingface_hub import hf_hub_download


def extract_verified(archive, entry, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    digest = hashlib.sha256()
    with archive.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if archive.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
        raise ValueError(f"Checksum or size mismatch: {entry['filename']}")
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.filename
                    or not path.parts or path.parts[0] != entry['directory']
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError('Unexpected archive member path.')
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive target is outside the output directory.')
        bundle.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--only', nargs='+', choices=['expression', 'metadata', 'knowledge'])
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent.parent / 'data_manifest.json').read_text())
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        if args.only and entry['directory'] not in args.only:
            continue
        archive = hf_hub_download(repo_id=manifest['repo_id'], repo_type='dataset',
            filename=entry['filename'], local_dir=destination / 'archives', token=False)
        extract_verified(archive, entry, destination)
        print(f"Ready: {destination / entry['directory']}")
    print('Set the input paths in your local configuration to the directories above.')


if __name__ == '__main__':
    main()
