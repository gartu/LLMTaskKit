import json
import re
import logging
from typing import Any, Union

from litellm import ModelResponse, CustomStreamWrapper
from LLMTaskKit.core.task import Task
from LLMTaskKit.core.llm import LLMConfig


class TaskExecutor:
    """
    Executes a Task by preparing prompts, invoking the LLM, and processing the response.
    """

    def __init__(self, default_llm: LLMConfig, verbose: bool = False):
        """
        Initializes the TaskExecutor.

        Args:
            default_llm (LLMConfig): The default LLM configuration to use.
            verbose (bool): Flag to enable verbose logging.
        """
        self.default_llm = default_llm
        self.verbose = verbose

    def execute(self, task: Task, context: dict) -> Any:
        """
        Executes the given task by constructing prompts and calling the LLM.
        
        Args:
            task (Task): The task to execute.
            context (dict): The context for prompt substitution.
        
        Returns:
            Any: The processed result from the LLM, optionally validated by a Pydantic model.
        """
        self.context = context

        if self.verbose:
            logging.info(f"Executing task {task.name}")

        # Construct prompts using Task methods
        system_prompt = task.get_system_prompt(self.context)
        prompt = task.get_prompt(self.context)

        if self.verbose:
            logging.info(f"System prompt: {system_prompt}")
            logging.info(f"User prompt: {prompt}")

        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

        is_thinking = "thinking" in task and task.thinking is not None
        is_output_format_forced = task.forced_output_format is not None
        
        assistant_prefill = ''
        if is_thinking:
            assistant_prefill = f"<thinking>{task.thinking}"
        elif is_output_format_forced:
            assistant_prefill = f"```{task.forced_output_format}\n"
        elif task.assistant_prefill is not None:
            assistant_prefill = task.assistant_prefill

        messages.append({"role": "assistant", "content": assistant_prefill})

        try:
            llm = self.default_llm if task.llm is None else task.llm
            response = llm.completion(messages=messages)
        except Exception as e:
            logging.error(f"Error calling LLM for task {task.name}: {e}")
            return None

        content = self._extract_response_content(response)
        content = f"{assistant_prefill if assistant_prefill else ''}{content}"

        if self.verbose:
            logging.info(f"Raw content: {content}")

        if is_thinking:
            content = self._remove_thinking(content)
            if is_output_format_forced:
                assistant_prefill = f"```{task.forced_output_format}\n"

        if assistant_prefill:
            if not content.startswith(assistant_prefill):
                content = f"{assistant_prefill}{content}"
            if is_output_format_forced:
                content = self._clean_response(content, task.forced_output_format)

        parsed_content: Any = content
        if task.output_pydantic is not None:
            try:
                json_content = json.loads(content)
                parsed_content = task.output_pydantic.model_validate(json_content)
            except Exception as ve:
                logging.error(f"Validation error for task {task.name}: {ve}")
        return parsed_content

    def _clean_response(self, content: str, output_format: str) -> str:
        """
        Cleans the response content by removing markdown or code block delimiters.
        
        Args:
            content (str): The raw response content.
        
        Returns:
            str: The cleaned content.
        """
        pattern = fr'^```(?:{output_format})?\n(.*?)```'
        match = re.search(pattern, content, flags=re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
            extra = content[match.end():].strip()
            if extra and self.verbose:
                logging.info("Response truncated: %s", extra)
            return cleaned
        else:
            # Si le délimiteur de fin n'est pas trouvé, on supprime uniquement celui du début.
            content = re.sub(fr'^```(?:{output_format})?\n', '', content, count=1)
            return content.strip()


    def _remove_thinking(self, text: str) -> str:
        return re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()

    def _extract_response_content(self, response: Union[ModelResponse, CustomStreamWrapper]) -> str:
        """
        Extracts the content from the LLM response.
        
        Args:
            response (Union[ModelResponse, CustomStreamWrapper]): The raw response from the LLM.
        
        Returns:
            str: The extracted content string.
        """
        try:
            content = response.choices[0].message.content.strip()
            return content
        except (AttributeError, IndexError) as e:
            logging.error("Error extracting response content: %s", e)
            return ""
