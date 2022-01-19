# https://github.com/pyinstaller/pyinstaller/issues/4400

from PyInstaller.utils.hooks import collect_all, collect_submodules


def hook(hook_api):
    packages = [
        'tensorflow',
        'keras',
        'sklearn'
    ]
    for package in packages:
        datas, binaries, hiddenimports = collect_all(package)
        hook_api.add_datas(datas)
        hook_api.add_binaries(binaries)
        hook_api.add_imports(*hiddenimports)
