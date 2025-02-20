from typing import Any, Dict, List

from LLMTaskKit.core.task import Task, TaskExecutor
from LLMTaskKit.core.llm import LLMConfig

TASK_RESULT_KEY = "TASK_RESULT"


class TaskChainExecutor:
    """
    Executes a chain of tasks sequentially, passing context between them.
    """

    def __init__(
        self,
        llm: LLMConfig,
        verbose: bool = False,
        step_by_step: bool = False,
    ) -> None:
        """
        Initializes the TaskChainExecutor.

        Args:
            llm (LLMConfig): The LLM configuration to use.
            verbose (bool): Flag to enable verbose logging.
        """
        self.result = None
        self.llm = llm
        self.verbose = verbose
        self.step_by_step = step_by_step

    def execute(self, tasks: List[Task], context: Dict[str, Any] = None) -> Any:
        """
        Executes each task in the chain sequentially, storing each result in the context.

        Args:
            tasks (List[Task]): A list of tasks to execute in order.
            context (Dict[str, Any], optional): The initial context for tasks.

        Returns:
            Any: The result of the last task in the chain.
        """
        self._init_context(context)

        task_executor = TaskExecutor(self.llm, verbose=self.verbose)
        for task in tasks:
            result = task_executor.execute(task, self.context)
            self.context[TASK_RESULT_KEY][task.name] = result

            if self.step_by_step:
                input("Press enter to continue")
                
        self.result = self.context[TASK_RESULT_KEY][tasks[-1].name]
        return self.result

    def _init_context(self, context: Dict[str, Any]):
        self.context: Dict[str, Any] = context if context is not None else {}
        if TASK_RESULT_KEY not in self.context:
            self.context[TASK_RESULT_KEY] = {}