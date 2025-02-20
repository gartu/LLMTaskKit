from typing import Any, Dict, List, Union
import logging

from pydantic import BaseModel, PrivateAttr


class RawTask(BaseModel):
    """
    Represents a raw task loaded from YAML with extra fields allowed.
    """
    name: str
    system_prompt: str

    # Preserve the key order as defined in the YAML
    _order: List[str] = PrivateAttr()

    class Config:
        extra = "allow"  # Allow additional fields

    @staticmethod
    def from_yaml(data: Union[Dict[str, Any], List[Any]]) -> Dict[str, "RawTask"]:
        """
        Constructs a dictionary of RawTask objects from YAML data.
        The YAML data can either be a list of tasks or a dictionary where keys are task names.
        
        Args:
            data (Union[Dict[str, Any], List[Any]]): The YAML data.
        
        Returns:
            Dict[str, RawTask]: A dictionary mapping task names to RawTask instances.
        
        Raises:
            ValueError: If a task in the list does not have a 'name' attribute.
        """
        tasks = {}
        if isinstance(data, list):
            for item in data:
                if "name" not in item:
                    raise ValueError("Each task must have a 'name' attribute.")
                raw_task = RawTask(**item)
                raw_task._order = list(item.keys())
                tasks[raw_task.name] = raw_task
        elif isinstance(data, dict):
            for name, item in data.items():
                item["name"] = name
                raw_task = RawTask(**item)
                raw_task._order = list(item.keys())
                tasks[name] = raw_task
        else:
            logging.error("Unrecognized YAML format for tasks.")
        return tasks
