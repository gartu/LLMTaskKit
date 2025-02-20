from typing import Any, Dict

from .prompt_builder import PromptBuilder, substitute_placeholders


class XmlPromptBuilder(PromptBuilder):
    """
    A PromptBuilder implementation that constructs prompts in an XML-like format.
    """

    def build_prompt(self, fields: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Builds an XML-like prompt from the given fields and context.
        
        Args:
            fields (Dict[str, Any]): The fields to include in the prompt.
            context (Dict[str, Any]): The context for placeholder substitution.
        
        Returns:
            str: The XML-like formatted prompt.
        """
        prompt_parts = []
        for key, value in fields.items():
            # If the value is a string, substitute placeholders
            if isinstance(value, str):
                substituted = substitute_placeholders(value, context)
                prompt_parts.append(f"<{key}>\n{substituted}\n</{key}>\n")
            # If the value is a list, process each item individually
            elif isinstance(value, list):
                if key.endswith("s") or key.endswith("List"):
                    # Determine the singular tag name for inner items
                    inner_tag = key[:-4] if key.endswith("List") else key[:-1]
                    inner_items = "".join(
                        f"<{inner_tag}>\n{substitute_placeholders(str(item), context)}\n</{inner_tag}>"
                        for item in value
                    )
                    prompt_parts.append(f"<{key}>\n{inner_items}\n</{key}>\n")
                else:
                    substituted = "\n".join(
                        substitute_placeholders(str(item), context) for item in value
                    )
                    prompt_parts.append(f"<{key}>\n{substituted}\n</{key}>\n")
            # For other types, convert to string and substitute placeholders
            else:
                substituted = substitute_placeholders(str(value), context)
                prompt_parts.append(f"<{key}>\n{substituted}\n</{key}>\n")
        return "".join(prompt_parts)
