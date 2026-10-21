"""Build an explicit code-only allowlist; never publish a workspace wholesale."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'release/skin-lab'
FILES = ['skin_generator.py', 'skin_generator_large.py', 'skin_generator_server.py', 'skin_tagger.py',
         'next_tag.py', 'main.py', 'catalog_server.py', 'tag_skins_folder.py', 'validate_skins_folder.py',
         'build_skins_sqlite.py', 'tags_top.py', 'start_generator_training.ps1', 'launch.py', 'start.ps1', 'start.sh',
         'download_models.py', 'model_hub.json', 'requirements.txt', '.gitignore', 'LICENSE', 'SUPPORT.md',
         'test_skin_generator.py', 'test_skin_generator_large.py', 'test_skin_generator_server.py',
         'test_skin_edit.py', 'test_next_tag.py', 'test_distribution.py']


def build():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    paths = [Path(name) for name in FILES if (ROOT / name).is_file()]
    for directory in ('generator_site', 'catalog_site', 'docs', '.github'):
        paths.extend(path.relative_to(ROOT) for path in (ROOT / directory).rglob('*') if path.is_file())
    paths.extend(Path('tools') / name for name in ('export_models.py', 'build_public_release.py', 'build_gallery.py'))
    for relative in paths:
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copyfile(ROOT / 'README_PUBLIC.md', DESTINATION / 'README.md')
    shutil.copyfile(ROOT / 'README_PUBLIC.md', DESTINATION / 'README_PUBLIC.md')
    print(f'Code release ready: {DESTINATION} ({len(paths) + 2} files)')


if __name__ == '__main__':
    build()
