# Prompt Fine-Tuner

Prompt Fine-Tuner is a Python project designed to elevate the effectiveness of prompts by guiding users through a structured, multi-step refinement process. By interacting with users to clarify objectives and gather key context, the tool iteratively transforms basic prompt drafts into highly optimized final prompts.

## Overview

- **Purpose:** Enhance the clarity, precision, and impact of prompts for large language models.
- **Functionality:** The project engages users with targeted questions, generates multiple draft versions, evaluates each iteration based on defined criteria, and refines the prompt to its optimal form—all through a seamless, interactive workflow.

## Impact on Prompt Quality

By systematically refining prompt drafts, Prompt Fine-Tuner:
- Eliminates ambiguity by asking clarifying questions.
- Offers diverse perspectives through multiple draft options.
- Incorporates detailed evaluations and feedback for continuous improvement.
- Produces final prompts that are coherent, actionable, and tailored to the intended use case.

## Setup Instructions

1. Ensure that a `.env` file is present at the root of the project with the `GEMINI_API_KEY` defined. You can obtain the API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Run the following command from the root of the project:
   ```sh
   poetry run python -m llmtaskkit.example.prompt_finetuner
   ```

