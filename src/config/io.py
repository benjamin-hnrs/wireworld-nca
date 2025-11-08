# config/io.py
import yaml
from dataclasses import asdict, is_dataclass
from typing import Any, Dict
from .run import RunCfg

def load_cfg(path: str) -> RunCfg:
    with open(path, "r") as f:
        # hack because i was silly and saved devices in the yaml
        yaml_text = f.read()

        import re
        yaml_text = re.sub(
            r'device:\s*!!python/object/apply:torch\.device\s*\n\s*-\s*(\w+)',
            r'device: "\1"',
            yaml_text
        )

        raw = yaml.safe_load(yaml_text) or {}
    return _merge_into_dataclass(RunCfg(), raw)

def _merge_into_dataclass(dc, updates: Dict[str, Any]):
    from dataclasses import fields, replace, is_dataclass
    kwargs = {}
    for f in fields(dc):
        if f.name in updates:
            if is_dataclass(getattr(dc, f.name)):
                kwargs[f.name] = _merge_into_dataclass(getattr(dc, f.name), updates[f.name])
            else:
                kwargs[f.name] = updates[f.name]
        else:
            kwargs[f.name] = getattr(dc, f.name)
    return replace(dc, **kwargs)
