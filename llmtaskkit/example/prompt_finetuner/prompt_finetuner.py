from dotenv import load_dotenv
import os
import logging
from llmtaskkit.core.llm import LLMConfig
from llmtaskkit.core.task import load_raw_tasks_from_yaml, Task, TaskExecutor
from llmtaskkit.prompt_chain.task_chain_executor import TaskChainExecutor
from pydantic import BaseModel
from typing import List, cast, Any

class Questions(BaseModel):
    questions: List[str]

class PromptDraft(BaseModel):
    prompt_number: int
    draft: str

class PromptDrafts(BaseModel):
    drafts: List[PromptDraft]

class PromptEvaluation(BaseModel):
    prompt_number: int
    feedback: str
    score: float

class PromptEvaluations(BaseModel):
    evaluations: List[PromptEvaluation]
    best_prompt: int
    
class RefinedPromptWithBrainstorm(BaseModel):
    brainstorm: str
    refined_prompt: str

class PromptFinetuner:

    def __init__(self):
        load_dotenv()
        gemini_api_key = os.getenv('GEMINI_API_KEY')

        self.llm = LLMConfig(api_key=gemini_api_key, model="gemini/gemini-2.0-flash-exp", temperature=0.8)

    def _callback(self, task_name: str, result: Any) -> None:
        logging.info(f"Task {task_name} completed with result: {result}")

    def exec(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        raw_tasks = load_raw_tasks_from_yaml("./llmtaskkit/example/prompt_finetuner/prompts_finetuner_tasks_fr.yaml")

        task_StretchPrompt = Task.from_raw(raw_tasks["StretchPrompt"], output_pydantic=Questions, forced_output_format="json")
        task_DraftPrompt = Task.from_raw(raw_tasks["DraftPrompt"], output_pydantic=PromptDrafts, forced_output_format="json")
        task_EvaluatePrompt = Task.from_raw(raw_tasks["EvaluatePrompt"], output_pydantic=PromptEvaluations, forced_output_format="json")
        task_PimpPrompt = Task.from_raw(raw_tasks["PimpPrompt"], output_pydantic=RefinedPromptWithBrainstorm, forced_output_format="json")
        task_ReviewPrompt = Task.from_raw(raw_tasks["ReviewPrompt"], forced_output_format="markdown")

        self.prompt = input("Enter your dummy prompt : ")
        context = { "prompt": self.prompt }

        executor = TaskExecutor(self.llm, verbose=False)
        questions = executor.execute(task_StretchPrompt, context)

        answers = []
        for question in cast(Questions, questions).questions:
            answer = input(f"{question} ")
            answers.append({"question": question, "answer": answer})

        context = {**executor.context, "questions": answers}

        tasks_chain = [task_DraftPrompt, task_EvaluatePrompt, task_PimpPrompt, task_ReviewPrompt]
        chain_executor = TaskChainExecutor(self.llm, verbose=False, callback=self._callback)
        chain_executor.execute(tasks_chain, context)

        logging.info("Result :")
        logging.info(chain_executor.result)
