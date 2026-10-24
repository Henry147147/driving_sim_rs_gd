"""Fetch only pinned CUDA compiler/runtime components for optional Windows/Linux GPU validation.

No driver or system installation is changed. NVIDIA archive SHA-256 values are
verified against the pinned release manifest before safe extraction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tarfile
import urllib.request
import zipfile
import shutil

BASE='https://developer.download.nvidia.com/compute/cuda/redist/'
RELEASE='13.2.0'
COMPONENTS=('cuda_nvcc','cuda_cudart','cuda_cccl','cuda_crt','libnvvm')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='target/cuda-13.2');a=p.parse_args()
    if platform.system() not in ('Linux','Windows') or platform.machine().lower() not in ('x86_64','amd64'):p.error('this helper supports Windows/Linux x86_64')
    platform_key='windows-x86_64' if platform.system()=='Windows' else 'linux-x86_64'
    root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=json.load(urllib.request.urlopen(BASE+f'redistrib_{RELEASE}.json'))
    for component in COMPONENTS:
        meta=manifest[component][platform_key];archive=root/(component+('.zip' if platform_key.startswith('windows') else '.tar.xz'))
        if not archive.exists():urllib.request.urlretrieve(BASE+meta['relative_path'],archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError(f'checksum mismatch: {archive}')
        if archive.suffix=='.zip':
            with zipfile.ZipFile(archive) as bundle:
                for item in bundle.infolist():
                    relative=Path(*Path(item.filename).parts[1:])
                    target=(root/relative).resolve()
                    if not target.is_relative_to(root):raise ValueError('unsafe archive path')
                    if item.is_dir():target.mkdir(parents=True,exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True,exist_ok=True)
                        with bundle.open(item) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
        else:
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    member.name='/'.join(member.name.split('/')[1:])
                    if member.name:tar.extract(member,root,filter='data')
        print(component,meta['sha256'],flush=True)
    (root/'redistribution-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Compiler: {root}/bin/nvcc. Link with -L{root}/lib.')
if __name__=='__main__':main()
