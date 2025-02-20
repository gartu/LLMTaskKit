import json
import re
import logging
import pytest
from pydantic import BaseModel

# Import the classes under test.
from LLMTaskKit.core.task import Task
from LLMTaskKit.core.llm import LLMConfig
from LLMTaskKit.core.prompt import XmlPromptBuilder

# ------------------------------------------------------------------------------
# Dummy Classes for Testing
# ------------------------------------------------------------------------------

class DummyLLMConfig(LLMConfig):
    """
    A dummy LLM configuration that mimics the behavior of the actual LLMConfig.
    It inherits from LLMConfig so that isinstance(instance, LLMConfig) returns True.
    """
    def __init__(self, response_content="default response", raise_exception=False, model="dummy", **kwargs):
        # Supply a default value for 'model' required by LLMConfig.
        self.model = model
        self.response_content = response_content
        self.raise_exception = raise_exception

    def completion(self, messages):
        if self.raise_exception:
            raise Exception("Fake LLM error")
        return DummyResponse(self.response_content)

class DummyResponseMessage:
    def __init__(self, content):
        self.content = content

class DummyResponseChoice:
    def __init__(self, message):
        self.message = message

class DummyResponse:
    def __init__(self, content):
        self.choices = [DummyResponseChoice(DummyResponseMessage(content))]

class DummyPromptBuilder:
    """
    A dummy prompt builder that returns a string representation of the fields and context.
    """
    def build_prompt(self, fields, context):
        return f"built_prompt: {fields} with context: {context}"

class DummyRawTask:
    """
    A dummy RawTask object used for testing the from_raw class method.
    """
    def __init__(self, data, order):
        self._data = data
        self._order = order

    def model_dump(self):
        return self._data

class DummyOutputModel(BaseModel):
    result: str

# ------------------------------------------------------------------------------
# Tests for the Task class
# ------------------------------------------------------------------------------

def test_system_prompt_validator():
    # system_prompt must be a string; passing a non-string should raise ValueError.
    with pytest.raises(ValueError, match="system_prompt must be a string."):
        Task(name="Test", system_prompt=123)

def test_llm_validator_instance(monkeypatch):
    # Passing an instance of DummyLLMConfig should be accepted as-is.
    monkeypatch.setattr("LLMTaskKit.core.llm.LLMConfig", DummyLLMConfig)
    dummy_llm = DummyLLMConfig(response_content="instance response", model="dummy")
    task = Task(name="Test", system_prompt="Prompt", llm=dummy_llm)
    # The validator should simply return the provided instance.
    assert task.llm is dummy_llm

def test_llm_validator_invalid():
    # Passing an invalid type for llm should raise a ValueError.
    with pytest.raises(ValueError, match="llm must be a configuration dictionary or an instance of LLMConfig."):
        Task(name="Test", system_prompt="Prompt", llm=123)

def test_output_pydantic_validator_valid():
    # Passing a valid Pydantic model class should be accepted.
    task = Task(name="Test", system_prompt="Prompt", output_pydantic=DummyOutputModel)
    assert task.output_pydantic is DummyOutputModel

def test_output_pydantic_validator_invalid():
    # Passing an invalid value (non-model) should raise a ValueError.
    with pytest.raises(ValueError, match="output_pydantic must be a Pydantic model"):
        Task(name="Test", system_prompt="Prompt", output_pydantic=123)

def test_init_conflict():
    # Both assistant_prefill and forced_output_format cannot be defined simultaneously.
    with pytest.raises(ValueError, match="Attributes assistant_prefill and forced_output_format cannot be both defined."):
        Task(name="Test", system_prompt="Prompt", assistant_prefill="prefill", forced_output_format="json")

def test_prompt_builder_override():
    # When a prompt_builder is provided in the constructor, it should override the default.
    custom_builder = DummyPromptBuilder()
    task = Task(name="Test", system_prompt="Prompt", prompt_builder=custom_builder)
    assert task._prompt_builder is custom_builder

def test_from_raw(monkeypatch):
    # Test the from_raw method by providing a DummyRawTask.
    data = {
        "name": "RawTask",
        "system_prompt": "Raw prompt",
        "llm": {"response_content": "raw llm response", "model": "dummy"},
    }
    order = ["name", "system_prompt", "custom_field"]
    raw_task = DummyRawTask(data, order)
    # Override with an extra field and a forced_output_format.
    task = Task.from_raw(raw_task, custom_field="value", forced_output_format="json")
    assert task.name == "RawTask"
    assert task.system_prompt == "Raw prompt"
    assert task.forced_output_format == "json"
    # Since extra fields are allowed, 'custom_field' should be set.
    assert getattr(task, "custom_field", None) == "value"
    assert task._order == order

def fake_substitute_placeholders(prompt, context):
    # Your fake implementation for testing.
    return prompt.format(**context)

@pytest.fixture
def builder(monkeypatch):
    # Patch substitute_placeholders in the xml_prompt_builder module
    monkeypatch.setattr("LLMTaskKit.core.prompt.xml_prompt_builder.substitute_placeholders", fake_substitute_placeholders)
    return XmlPromptBuilder()

def test_get_system_prompt(monkeypatch):
    task = Task(name="Test", system_prompt="Hello {{name}}")
    context = {"name": "World"}
    prompt = task.get_system_prompt(context)
    assert prompt == "Hello World"

def test_get_prompt():
    # Create a Task with extra fields and a custom order.
    custom_builder = DummyPromptBuilder()
    task = Task(name="Test", system_prompt="Prompt")
    # Dynamically add extra fields; note that "thinking" is skipped in get_prompt.
    setattr(task, "param", "value")
    setattr(task, "thinking", "should be skipped")
    # Define the order so that only "param" is used.
    task._order = ["name", "system_prompt", "param", "thinking", "non_existing"]
    # Override the default prompt builder.
    task._prompt_builder = custom_builder
    context = {"key": "val"}
    result = task.get_prompt(context)
    expected_fields = {"param": "value"}
    expected = custom_builder.build_prompt(expected_fields, context)
    assert result == expected
