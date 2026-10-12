"""Check installed modules and bundled resources from any working directory."""
import importlib
import importlib.resources
import json
import pkgutil
import cellcanon


def main():
    modules = [item.name for item in pkgutil.walk_packages(cellcanon.__path__, cellcanon.__name__ + '.')]
    for name in modules:
        importlib.import_module(name)
    root = importlib.resources.files('cellcanon')
    for name in ('split.json', 'genes.json', 'update.json', 'update_schedule.json'):
        json.loads(root.joinpath('protocols', name).read_text())
    assert root.joinpath('resources', 'sciplex3_raw_stream.cpp').is_file()
    print(json.dumps({'status': 'PASS',
                      'modules_imported': len(modules), 'protocol_files': 4,
                      'preprocessing_resource': True}))


if __name__ == '__main__':
    main()
