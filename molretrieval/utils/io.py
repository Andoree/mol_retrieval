from typing import Dict


def load_config_dict(config_path: str, sep=' : ') -> Dict[str, str]:
    cfg_dict: Dict[str, str] = {}
    with open(config_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split(sep)
            assert len(attrs) == 2
            key, val = attrs
            cfg_dict[key] = val
    return cfg_dict
