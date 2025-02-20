from typing import Any, Dict
from . import PromptBuilder, substitute_placeholders

class MarkdownPromptBuilder(PromptBuilder):
    def build_prompt(self, fields: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt_parts = []
        for key, value in fields.items():
            # Si la valeur est une chaîne de caractères
            if isinstance(value, str):
                substituted = substitute_placeholders(value, context)
                prompt_parts.append(f"### {key.capitalize()}\n{substituted}\n")
            
            # Si la valeur est une liste
            elif isinstance(value, list):
                if key.endswith("s") or key.endswith("List"):
                    # Ajouter un titre pour la liste
                    prompt_parts.append(f"### {key.capitalize()}\n")
                    # Formater chaque élément comme une puce Markdown
                    for item in value:
                        substituted_item = substitute_placeholders(str(item), context)
                        prompt_parts.append(f"- {substituted_item}\n")
                    prompt_parts.append("\n")  # Ajouter une ligne vide après la liste
                else:
                    # Si la clé ne se termine pas par "s" ou "List", traiter comme une simple liste
                    substituted_items = "\n".join(
                        f"- {substitute_placeholders(str(item), context)}" for item in value
                    )
                    prompt_parts.append(f"### {key.capitalize()}\n{substituted_items}\n")
            
            # Pour les autres types, conversion en chaîne et substitution
            else:
                substituted = substitute_placeholders(str(value), context)
                prompt_parts.append(f"### {key.capitalize()}\n{substituted}\n")
        
        return "".join(prompt_parts)