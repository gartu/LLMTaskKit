
import logging
import pytest
from pydantic import BaseModel

# Import the classes under test
from LLMTaskKit.core.task import Task, TaskExecutor

class DummyLLMConfig:
    """
    A dummy LLM configuration that mimics the behavior of the actual LLMConfig.
    """
    def __init__(self, response_content="default response", raise_exception=False, **kwargs):
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

class DummyOutputModel(BaseModel):
    result: str

# ------------------------------------------------------------------------------
# Tests for the TaskExecutor class
# ------------------------------------------------------------------------------

@pytest.fixture
def dummy_executor(monkeypatch):
    # Create a dummy TaskExecutor with a default LLM (using our dummy implementation).
    monkeypatch.setattr("LLMTaskKit.core.llm.LLMConfig", DummyLLMConfig)
    default_llm = DummyLLMConfig(response_content="default response", model="dummy")
    return TaskExecutor(default_llm=default_llm, verbose=True)

def test_execute_basic(dummy_executor):
    # Test a basic execution with no special flags.
    task = Task(name="BasicTask", system_prompt="System {var}")
    # Override prompt construction methods for controlled outputs.
    task.get_system_prompt = lambda ctx: f"System prompt with {ctx.get('var')}"
    task.get_prompt = lambda ctx: f"User prompt with {ctx.get('var')}"
    context = {"var": "value"}
    result = dummy_executor.execute(task, context)
    # Since no assistant_prefill is set, the result should be the dummy LLM response.
    assert result == "default response"

def test_execute_with_forced_format(dummy_executor):
    # Test execution when forced_output_format is provided.
    task = Task(name="ForcedTask", system_prompt="System", forced_output_format="json")
    task.get_system_prompt = lambda ctx: "System prompt"
    task.get_prompt = lambda ctx: "User prompt"
    context = {}
    # Create a dummy LLM that returns valid JSON without markdown delimiters.
    # The assistant_prefill ("```json\n") will be prepended and then cleaned.
    forced_content = "{\"result\": \"ok\"}"
    dummy_llm = DummyLLMConfig(response_content=forced_content, model="dummy")
    task.llm = dummy_llm
    result = dummy_executor.execute(task, context)
    # _clean_response should extract the JSON content.
    assert result == "{\"result\": \"ok\"}"

def test_execute_with_thinking(dummy_executor):
    # Test execution when a 'thinking' attribute is set.
    task = Task(name="ThinkingTask", system_prompt="System")
    # For testing, we set thinking to a value that includes a closing tag.
    setattr(task, "thinking", "processing</thinking>")
    task.get_system_prompt = lambda ctx: "System prompt"
    task.get_prompt = lambda ctx: "User prompt"
    context = {}
    # Dummy LLM returns a response that includes the thinking tag.
    response_content = "<thinking>processing</thinking>result"
    dummy_llm = DummyLLMConfig(response_content=response_content, model="dummy")
    task.llm = dummy_llm
    result = dummy_executor.execute(task, context)
    # The _remove_thinking method removes the thinking portion.
    # Since the code prepends the thinking tag, we expect it to start with "<thinking>".
    assert result.startswith("<thinking>")
    assert "result" in result

def test_execute_with_assistant_prefill(dummy_executor):
    # Test execution when assistant_prefill is provided.
    task = Task(name="PrefillTask", system_prompt="System", assistant_prefill="Prefilled answer")
    task.get_system_prompt = lambda ctx: "System prompt"
    task.get_prompt = lambda ctx: "User prompt"
    context = {}
    dummy_llm = DummyLLMConfig(response_content="response content", model="dummy")
    task.llm = dummy_llm
    result = dummy_executor.execute(task, context)
    # The result should start with the assistant_prefill and include the LLM response.
    assert result.startswith("Prefilled answer")
    assert "response content" in result

def test_execute_with_output_pydantic_valid(dummy_executor):
    # Test execution when output_pydantic is provided and the LLM returns valid JSON.
    task = Task(name="PydanticTask", system_prompt="System", output_pydantic=DummyOutputModel)
    task.get_system_prompt = lambda ctx: "System prompt"
    task.get_prompt = lambda ctx: "User prompt"
    context = {}
    json_response = "{\"result\": \"success\"}"
    dummy_llm = DummyLLMConfig(response_content=json_response, model="dummy")
    task.llm = dummy_llm
    result = dummy_executor.execute(task, context)
    # The result should be a validated DummyOutputModel instance.
    assert isinstance(result, DummyOutputModel)
    assert result.result == "success"

def test_execute_with_output_pydantic_invalid(dummy_executor, caplog):
    # Test execution when output_pydantic is provided but the LLM returns invalid JSON.
    task = Task(name="InvalidPydanticTask", system_prompt="System", output_pydantic=DummyOutputModel)
    task.get_system_prompt = lambda ctx: "System prompt"
    task.get_prompt = lambda ctx: "User prompt"
    context = {}
    invalid_json = "not a json"
    dummy_llm = DummyLLMConfig(response_content=invalid_json, model="dummy")
    task.llm = dummy_llm
    with caplog.at_level(logging.ERROR):
        result = dummy_executor.execute(task, context)
        assert "Validation error" in caplog.text
    # In case of validation failure, the raw response string is returned.
    assert result == invalid_json

def test_extract_response_content_error(dummy_executor, caplog):
    # Test _extract_response_content with a response missing the expected attributes.
    faulty_response = object()  # Does not have a 'choices' attribute.
    content = dummy_executor._extract_response_content(faulty_response)
    assert content == ""
    # Check that an error message was logged.
    assert "Error extracting response content" in caplog.text

def test_clean_response_no_match(dummy_executor):
    # Test _clean_response on content that does not fully match the markdown pattern.
    content = "```json\n{\"result\": \"ok\""
    cleaned = dummy_executor._clean_response(content, "json")
    # It should remove the starting delimiter and strip the string.
    assert cleaned == "{\"result\": \"ok\""

def test_llm_exception(dummy_executor, caplog):
    # Test that if llm.completion raises an exception, execute returns None.
    task = Task(name="ExceptionTask", system_prompt="System")
    task.get_system_prompt = lambda ctx: "System prompt"
    task.get_prompt = lambda ctx: "User prompt"
    context = {}
    dummy_llm = DummyLLMConfig(raise_exception=True, model="dummy")
    task.llm = dummy_llm
    with caplog.at_level(logging.ERROR):
        result = dummy_executor.execute(task, context)
        assert result is None
        assert "Error calling LLM for task" in caplog.text