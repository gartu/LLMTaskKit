from __future__ import annotations
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, field_validator, ConfigDict, PrivateAttr

from LLMTaskKit.core.llm import LLMConfig
from LLMTaskKit.core.task import RawTask
from LLMTaskKit.core.prompt import PromptBuilder, substitute_placeholders, XmlPromptBuilder


class Task(BaseModel):
    """
    Represents a task with a name, system prompt, LLM configuration,
    and optional Pydantic model for output validation.
    """
    name: str
    system_prompt: str
    llm: Optional[LLMConfig] = None
    output_pydantic: Optional[Type[BaseModel]] = None
    forced_output_format: Optional[str] = None
    assistant_prefill: Optional[str] = None

    # Private attribute for the prompt builder (not validated by Pydantic)
    _prompt_builder: PromptBuilder = PrivateAttr(default_factory=XmlPromptBuilder)
    # Private attribute to preserve the key order as defined in the YAML file
    _order: List[str] = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @field_validator("system_prompt", mode="before")
    @classmethod
    def validate_system_prompt(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("system_prompt must be a string.")
        return value

    @field_validator("llm", mode="before")
    @classmethod
    def validate_llm(cls, value: Any) -> Optional[LLMConfig]:
        if value is None:
            return None
        if isinstance(value, dict):
            return LLMConfig(**value)
        if isinstance(value, LLMConfig):
            return value
        raise ValueError("llm must be a configuration dictionary or an instance of LLMConfig.")

    @field_validator("output_pydantic", mode="before")
    @classmethod
    def validate_output_pydantic(cls, value: Any) -> Optional[Type[BaseModel]]:
        if value is None:
            return None
        if isinstance(value, type) and issubclass(value, BaseModel):
            return value
        raise ValueError("output_pydantic must be a Pydantic model (a class inheriting from BaseModel).")

    def __init__(self, *, prompt_builder: Optional[PromptBuilder] = None, **data: Any) -> None:
        """
        Initializes a Task instance with optional prompt_builder.
        
        Args:
            prompt_builder (Optional[PromptBuilder]): An optional prompt builder instance.
            **data: Additional task data.
        """
        super().__init__(**data)
        if self.assistant_prefill is not None and self.forced_output_format is not None:
            raise ValueError("Attributes assistant_prefill and forced_output_format cannot be both defined.")
        if prompt_builder is not None:
            self._prompt_builder = prompt_builder

    @classmethod
    def from_raw(
        cls,
        raw_task: RawTask,
        *,
        llm: Optional[LLMConfig] = None,
        output_pydantic: Optional[Type[BaseModel]] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        forced_output_format: Optional[str] = None,
        assistant_prefill: Optional[str] = None,
        **overrides: Any
    ) -> Task:
        """
        Constructs a Task instance from a RawTask, taking into account any explicit
        parameters for 'llm', 'output_pydantic', and 'prompt_builder'.
        
        Args:
            raw_task (RawTask): The raw task object.
            llm (Optional[LLMConfig]): An optional LLM configuration.
            output_pydantic (Optional[Type[BaseModel]]): An optional Pydantic model class for output validation.
            prompt_builder (Optional[PromptBuilder]): An optional prompt builder instance.
            forced_output_format (Optional[str]): An optional output format used to prefill assistant answer. Use 'json' for pydantic output.
            assistant_prefill (Optional[str]): An optional string to prefill assistant answer. The prefill string is sending back as the whole answer. 
            **overrides: Additional keyword arguments to override raw task data.
        
        Returns:
            Task: A new Task instance.
        """
        data = raw_task.model_dump()
        for key, value in (
            ("llm", llm),
            ("output_pydantic", output_pydantic),
            ("prompt_builder", prompt_builder),
            ("forced_output_format", forced_output_format),
            ("assistant_prefill", assistant_prefill),
        ):
            if value is not None:
                data[key] = value
        data.update(overrides)
        task = cls(**data)
        task._order = raw_task._order
        return task

    def get_system_prompt(self, context: Dict[str, Any]) -> str:
        """
        Constructs the system prompt by replacing dynamic keys using the provided context.
        
        Args:
            context (Dict[str, Any]): The context for placeholder substitution.
        
        Returns:
            str: The system prompt with placeholders substituted.
        """
        return substitute_placeholders(self.system_prompt, context)

    def get_prompt(self, context: Dict[str, Any]) -> str:
        """
        Constructs the user prompt by filtering out system-level keys and
        delegating the construction of the remaining fields to the prompt builder.
        
        Args:
            context (Dict[str, Any]): The context for placeholder substitution.
        
        Returns:
            str: The fully constructed prompt.
        """
        fields: Dict[str, Any] = {}
        for key in self._order:
            if key in ("name", "system_prompt", "llm", "output_pydantic", "thinking"):
                continue
            value = getattr(self, key, None)
            if value is not None:
                fields[key] = value
        if self.output_pydantic is not None:
            fields["output_pydantic"] = self.output_pydantic.model_json_schema()
        return self._prompt_builder.build_prompt(fields, context)
