from dotenv import load_dotenv
import os
import logging
from llmtaskkit.core.llm import LLMConfig
from llmtaskkit.core.task import load_raw_tasks_from_yaml, Task
from llmtaskkit.prompt_chain.task_chain_executor import TaskChainExecutor
from pydantic import BaseModel
from typing import List, cast, Any
import json

class SpeakerKey(BaseModel):
    speaker_key: str
    role: str
    main_skill: List[str]
    confidence_percent: int
    supposed_name: str
    
class RawSpeakers(BaseModel):
    speakers: List[SpeakerKey]
    
class Speaker(BaseModel):
    speaker: str
    role: str
    name: str
    
class Speakers(BaseModel):
    speakers: List[Speaker]

class KeyPoint(BaseModel):
    speaker: str
    contents: List[str]

class Topic(BaseModel):
    title: str
    key_points: List[KeyPoint]

class TranscriptionSummary(BaseModel):
    topics: List[Topic]

class Action(BaseModel):
    action: str
    description: str
    difficulty: str # Literal["HARD", "MEDIUM", "EASY", "UNKNOWN"]
    impact: str # Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
    deadline: str

class SpeakerActions(BaseModel):
    speaker: str
    actions: List[Action]
    
class SpeakersActions(BaseModel):
    speakers: List[SpeakerActions]


class SubjectRisksAndStakes(BaseModel):
    subject: str
    stakes: List[str]
    risks: List[str]

class RisksAndStakes(BaseModel):
    risks_and_stakes: List[SubjectRisksAndStakes]

class SubjectSummary(BaseModel):
    title: str
    summary: str
    decisions: List[str]

class Summary(BaseModel):
    summaries: List[SubjectSummary]


# https://github.com/anthropics/courses/blob/master/real_world_prompting/04_call_summarizer.ipynb
class MeetingSummary():
    def __init__(self):
        load_dotenv()
        gemini_api_key = os.getenv('GEMINI_API_KEY')

        # gemini-2.0-pro-exp-02-05
        self.llm = LLMConfig(api_key=gemini_api_key, model="gemini/gemini-2.0-flash-exp", temperature=0.2)

        # Gladia transcription
        # [
        #    {
        #        "speaker": "speaker_00",
        #        "text": "Hello !"
        #    },
        #    {
        #        "speaker": "speaker_01",
        #        "text": "Hi"
        #    },
        # ]
        transcription_path = "./assets/transcription.json"
        
        if os.path.exists(transcription_path):
            with open(transcription_path, "r", encoding="utf-8") as f:
                self.transcription = json.load(f)
        else:
            raise FileNotFoundError("Le traitement audio est désactivé et le fichier 'transcription.json' est introuvable.")
    
    def _update_rawspeakers_with_names(self, rawspeakers: RawSpeakers) -> RawSpeakers:
        for speaker in rawspeakers.speakers:
            # Affichage clair du rôle et des compétences
            role_info = f"{speaker.speaker_key}\nRôle: {speaker.role} | Compétences: {', '.join(speaker.main_skill)} | Nom supposé : {speaker.supposed_name}"
            print(role_info)
            
            # Demande de saisie du nom correspondant à ce rôle/compétence
            while True:
                nom = input("Veuillez entrer le nom de la personne pour ce rôle/compétence : ").strip()
                # Vérifier que l'entrée est bien une chaîne non vide
                if isinstance(nom, str) and nom != "":
                    speaker.supposed_name = nom
                    break  # Sortir de la boucle une fois une saisie valide obtenue
                else:
                    print("Erreur : vous devez fournir un nom valide. Veuillez réessayer.")
        
        return rawspeakers

    def _callback(self, task_name: str, result: Any) -> None:
        logging.info(f"Task {task_name} completed with result: {result}")

    def exec(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        initial_context = { "transcription": self.transcription }

        raw_tasks = load_raw_tasks_from_yaml("./llmtaskkit/example/meeting_summary/meeting_summary_tasks_fr.yaml")

        task_TopicExtraction = Task.from_raw(raw_tasks["TopicExtraction"], output_pydantic=TranscriptionSummary, forced_output_format="json")
        task_RolesExtraction = Task.from_raw(raw_tasks["RolesExtraction"], output_pydantic=RawSpeakers, forced_output_format="json")
                
        executor = TaskChainExecutor(self.llm, verbose=True, callback=self._callback)
        result = executor.execute([task_TopicExtraction, task_RolesExtraction], initial_context)

        initial_context = {**executor.context}

        raw_speakers = self._update_rawspeakers_with_names(result)
        speaker_by_key = {}
        for speaker in raw_speakers.speakers:
            speaker_by_key[speaker.speaker_key] = speaker.supposed_name
        
        # update speakers names
        for topic in cast(TranscriptionSummary, initial_context["TASK_RESULT"]["TopicExtraction"]).topics:
            for key_point in topic.key_points:
                key_point.speaker = speaker_by_key[key_point.speaker]

        task_RisksIdentification = Task.from_raw(raw_tasks["RisksIdentification"], output_pydantic=RisksAndStakes, forced_output_format="json")
        task_SummaryCreation = Task.from_raw(raw_tasks["SummaryCreation"], output_pydantic=Summary, forced_output_format="json")
        task_ActionsDefinition = Task.from_raw(raw_tasks["ActionsDefinition"], output_pydantic=SpeakersActions, forced_output_format="json")
        task_MeetingReportCreation = Task.from_raw(raw_tasks["MeetingReportCreation"], forced_output_format="markdown")
        task_MeetingReportReview = Task.from_raw(raw_tasks["MeetingReportReview"], forced_output_format="markdown")

        tasks_chain = [task_RisksIdentification, task_SummaryCreation, task_ActionsDefinition, task_MeetingReportCreation, task_MeetingReportReview]

        executor = TaskChainExecutor(self.llm, verbose=True, callback=self._callback)
        result = executor.execute(tasks_chain, initial_context)

        logging.info("Result :")
        logging.info(result)
