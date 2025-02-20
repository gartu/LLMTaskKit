# LLMTaskKit/core/task/task_loader.py
import yaml
from LLMTaskKit.core.task import RawTask

def load_raw_tasks_from_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RawTask.from_yaml(data)
