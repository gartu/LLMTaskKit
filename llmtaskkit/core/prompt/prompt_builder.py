from abc import ABC, abstractmethod
import re
import logging
from typing import Any, Dict


def substitute_placeholders(template: str, context: Dict[str, Any]) -> str:
    """
    Replaces placeholders of the form {{ expression }} with their corresponding values from the context.

    Args:
        template (str): The string template containing placeholders.
        context (Dict[str, Any]): The context dictionary for substitution.

    Returns:
        str: The template with placeholders replaced.
    """
    if not template:
        return ""

    def replacer(match: re.Match) -> str:
        expr = match.group(1).strip()
        try:
            parts = expr.split('.')
            val = context
            for part in parts:
                val = val[part]
            return str(val)
        except Exception as e:
            logging.warning(f"Unable to bind expression '{expr}': {e}")
            return match.group(0)

    return re.sub(r"\{\{(.*?)\}\}", replacer, template)


class PromptBuilder(ABC):
    """
    Abstract base class for building prompts from given fields and context.
    """

    @abstractmethod
    def build_prompt(self, fields: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Constructs the prompt from a dictionary of fields (already filtered)
        and a context.
        
        Args:
            fields (Dict[str, Any]): The fields to include in the prompt.
            context (Dict[str, Any]): The context for placeholder substitution.
        
        Returns:
            str: The constructed prompt.
        """
        pass
