from __future__ import annotations

import json
from typing import Union

from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish, OutputParserException


class ConvoOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            cleaned_output = text.strip()
            if "```json" in cleaned_output:
                _, cleaned_output = cleaned_output.split("```json")
            if "```" in cleaned_output:
                cleaned_output, _ = cleaned_output.split("```")
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[len("```json") :]
            if cleaned_output.startswith("```"):
                cleaned_output = cleaned_output[len("```") :]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[: -len("```")]
            cleaned_output = cleaned_output.strip()

            # response = json.loads(cleaned_output)

            try:
                response = json.loads(cleaned_output)
            except json.JSONDecodeError:
                response = {}
                cleaned_output = cleaned_output.replace('\n', '')

                try:
                    response["action"] = cleaned_output.split('"action": "', 1)[1].split('",', 1)[0].strip()
                    response["action_input"] = cleaned_output.split('"action_input": "', 1)[1].split('"}', 1)[0].strip()
                except IndexError:

                    raise ValueError(
                        "Invalid input format. Unable to extract 'action' and 'action_input' from the text.")

            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                return AgentAction(action, action_input, text)
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    @property
    def _type(self) -> str:
        return "conversational_chat"
