"""Portable first-run setup: Python venv, CPU/CUDA dependencies, models and local UI."""
from __future__ import annotations
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command):
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def choose_backend(requested):
    if requested != 'auto':
        return requested
    if platform.system() != 'Darwin' and shutil.which('nvidia-smi'):
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0 and 'GPU' in result.stdout:
            return 'cuda'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--repo', help='Hugging Face model repository override')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--models', default='all')
    parser.add_argument('--skip-install', action='store_true')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check installation and models, then exit')
    args = parser.parse_args()
    if sys.version_info[:2] < (3, 10) or sys.version_info[:2] > (3, 13):
        parser.error('Use 64-bit Python 3.10-3.13 (3.12 recommended).')
    if sys.maxsize <= 2**32:
        parser.error('64-bit Python is required.')
    backend = choose_backend(args.backend)
    if backend == 'cuda' and platform.system() == 'Darwin':
        parser.error('CUDA is available only on Linux/Windows with an NVIDIA GPU. Use --backend cpu.')
    environment = ROOT / '.venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print('Creating .venv ...', flush=True)
        venv.create(environment, with_pip=True)
    stamp = environment / 'skin-lab-setup.json'
    desired = {'torch': '2.7.1', 'torchvision': '0.22.1', 'backend': backend,
               'requirements': (ROOT / 'requirements.txt').read_text(encoding='utf-8')}
    previous = json.loads(stamp.read_text()) if stamp.exists() else None
    if not args.skip_install and previous != desired:
        print(f'Installing dependencies ({backend}) ...', flush=True)
        command = [python, '-m', 'pip', 'install', 'torch==2.7.1', 'torchvision==0.22.1']
        if platform.system() != 'Darwin':
            command += ['--index-url', 'https://download.pytorch.org/whl/' + ('cu126' if backend == 'cuda' else 'cpu')]
        if previous and previous.get('backend') != backend:
            command += ['--force-reinstall']
        run(command)
        run([python, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        stamp.write_text(json.dumps(desired), encoding='utf-8')
    if not args.skip_download:
        command = [python, 'download_models.py', '--models', args.models]
        if args.repo:
            command += ['--repo', args.repo]
        run(command)
    run([python, '-c', 'import torch, torchvision; from pathlib import Path; '
         'required=["generator_large_new_dataset/last.pt","generator_large_new_dataset/dataset.json","next_tag/model.pt","skin_tagger.pt"]; '
         'missing=[name for name in required if not (Path("checkpoints")/name).is_file()]; '
         'assert not missing, "Missing models: " + str(missing); '
         'print("Ready. CUDA available:",torch.cuda.is_available())'])
    if args.check:
        return
    url = f'http://127.0.0.1:{args.port}'
    print(f'Open {url} in your browser. Ctrl+C stops the server.', flush=True)
    if not args.no_browser:
        import threading
        threading.Timer(2, lambda: webbrowser.open(url)).start()
    run([python, 'skin_generator_server.py', '--port', str(args.port), '--device',
         'auto' if args.backend == 'auto' else args.backend])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as error:
        raise SystemExit(f'Command failed (exit {error.returncode}). See the message above.')
