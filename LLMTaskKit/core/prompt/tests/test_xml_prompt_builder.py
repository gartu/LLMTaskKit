import pytest

# Adaptez l'import ci-dessous selon la structure de votre projet.
from LLMTaskKit.core.prompt.xml_prompt_builder import XmlPromptBuilder

# Fonction factice pour simuler substitute_placeholders.
def fake_substitute_placeholders(text, context):
    """
    Retourne simplement le texte d'origine auquel est ajouté un suffixe (défini dans context ou '_s' par défaut).
    """
    suffix = context.get("suffix", "_s")
    return f"{text}{suffix}"

# Fixture pour créer une instance de XmlPromptBuilder avec la fonction de substitution patchée.
@pytest.fixture
def builder(monkeypatch):
    # Patch de la fonction substitute_placeholders dans le module xml_prompt_builder.
    monkeypatch.setattr("LLMTaskKit.core.prompt.xml_prompt_builder.substitute_placeholders", fake_substitute_placeholders)
    return XmlPromptBuilder()

def test_build_prompt_with_string(builder):
    """
    Teste la construction d'un prompt lorsqu'une valeur de type string est fournie.
    """
    fields = {"title": "Hello, {name}!"}
    context = {"name": "World", "suffix": "_sub"}
    expected = "<title>\nHello, {name}!_sub\n</title>\n"
    result = builder.build_prompt(fields, context)
    assert result == expected

def test_build_prompt_with_list_plural(builder):
    """
    Teste la construction d'un prompt pour une liste avec une clé se terminant par 's'.
    Le tag interne doit être le nom au singulier (clé sans le dernier caractère).
    """
    fields = {"items": ["a", "b"]}
    context = {"suffix": "_x"}
    # Pour 'items', le tag interne est 'item'
    inner = "<item>\na_x\n</item><item>\nb_x\n</item>"
    expected = f"<items>\n{inner}\n</items>\n"
    result = builder.build_prompt(fields, context)
    assert result == expected

def test_build_prompt_with_list_list(builder):
    """
    Teste la construction d'un prompt pour une liste avec une clé se terminant par 'List'.
    Le tag interne est obtenu en retirant les 4 derniers caractères.
    """
    fields = {"namesList": ["Alice", "Bob"]}
    context = {"suffix": "_y"}
    # Pour 'namesList', inner_tag = "namesList"[:-4] = "names"
    inner = "<names>\nAlice_y\n</names><names>\nBob_y\n</names>"
    expected = f"<namesList>\n{inner}\n</namesList>\n"
    result = builder.build_prompt(fields, context)
    assert result == expected

def test_build_prompt_with_list_non_special(builder):
    """
    Teste le cas où la valeur est une liste et que la clé ne se termine ni par 's' ni par 'List'.
    La liste est alors jointe avec des retours à la ligne.
    """
    fields = {"data": [1, 2]}
    context = {"suffix": "!"}
    # Chaque élément est converti en string et la substitution ajoute "!".
    substituted = "1!"+ "\n" + "2!"
    expected = f"<data>\n{substituted}\n</data>\n"
    result = builder.build_prompt(fields, context)
    assert result == expected

def test_build_prompt_with_non_string(builder):
    """
    Teste la construction d'un prompt pour une valeur non-string (ex. un entier).
    La valeur est convertie en string puis traitée.
    """
    fields = {"age": 30}
    context = {"suffix": "?"}
    expected = "<age>\n30?\n</age>\n"
    result = builder.build_prompt(fields, context)
    assert result == expected

def test_build_prompt_with_multiple_fields(builder):
    """
    Teste la construction d'un prompt avec plusieurs champs de types différents.
    """
    fields = {
        "title": "Title",
        "items": ["x", "y"],
        "data": [100, 200],
        "flag": True
    }
    context = {"suffix": "_test"}
    # Construction attendue pour chaque champ :
    # - title: simple string.
    title_part = "<title>\nTitle_test\n</title>\n"
    # - items: clé se terminant par 's' => inner tag 'item'
    inner_items = "<item>\nx_test\n</item><item>\ny_test\n</item>"
    items_part = f"<items>\n{inner_items}\n</items>\n"
    # - data: clé ne se terminant pas par 's' ni 'List' => join avec '\n'
    data_sub = "100_test\n200_test"
    data_part = f"<data>\n{data_sub}\n</data>\n"
    # - flag: valeur non-string convertie en string ("True")
    flag_part = "<flag>\nTrue_test\n</flag>\n"
    
    expected = title_part + items_part + data_part + flag_part
    result = builder.build_prompt(fields, context)
    assert result
