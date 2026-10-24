"""Fetch only pinned CUDA compiler/runtime components for optional Linux GPU validation.

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

BASE='https://developer.download.nvidia.com/compute/cuda/redist/'
RELEASE='13.2.0'
COMPONENTS=('cuda_nvcc','cuda_cudart','cuda_cccl','cuda_crt','libnvvm')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='target/cuda-13.2');a=p.parse_args()
    if platform.system()!='Linux' or platform.machine()!='x86_64':p.error('this helper supports Linux x86_64 (including WSL)')
    root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
    manifest=json.load(urllib.request.urlopen(BASE+f'redistrib_{RELEASE}.json'))
    for component in COMPONENTS:
        meta=manifest[component]['linux-x86_64'];archive=root/(component+'.tar.xz')
        if not archive.exists():urllib.request.urlretrieve(BASE+meta['relative_path'],archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError(f'checksum mismatch: {archive}')
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                member.name='/'.join(member.name.split('/')[1:])
                if member.name:tar.extract(member,root,filter='data')
        print(component,meta['sha256'],flush=True)
    (root/'redistribution-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Compiler: {root}/bin/nvcc. Link with -L{root}/lib.')
if __name__=='__main__':main()
