# LLM-driven agents

""" .env contains OPENAI_API_KEY  """

from  __future__ import annotations

from dotenv import load_dotenv
import json, os
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI                                     # GPT-4 wrapper
from langchain_core.prompts import PromptTemplate                           # Prompt building
from langchain.output_parsers import StructuredOutputParser, ResponseSchema # JSON parsing

load_dotenv()
print("API KEY LOADED:", os.getenv("OPENAI_API_KEY"))

# Helper LLM
LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Ontologist
class OntologistAgent:
    """ hypothesis: str, mechanism: str, assumptions: [str] """
    def __init__(self):
        self.schemas = [
            ResponseSchema(name="hypothesis",  description="One concise sentence"),
            ResponseSchema(name="mechanism",   description="1-2 sentences, linking each edge"),
            ResponseSchema(name="assumptions", description="list of explicit assumptions")
        ]
        self.parser = StructuredOutputParser.from_response_schemas(self.schemas)
        self.prompt = PromptTemplate(
            template=(
                "You are an expert ontologist.\n"
                "Given a knowledge-graph path, produce a causal research hypothesis.\n\n"
                "PATH:\n{path_str}\n\n"
                "Return JSON ONLY with the following keys:\n"
                "{format_instructions}"
            ),
            input_variables=["path_str"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()} # self.schema
        )

    def run(self, path: List[Dict[str, str]]) -> Dict[str, Any]:
        triples = [f"{s['head']} -[{s['relation']}]-> {s['tail']}" for s in path]
        path_str = " | ".join(triples)
        prompt_msg = self.prompt.format(path_str=path_str)
        llm_out = LLM.invoke(prompt_msg).content
        return self.parser.parse(llm_out)

class DomainExpertAgent:
    """ domain: str, elaboration: str, references: [str] """
    def __init__(self, domain: str):
        self.domain = domain
        self.schemas = [
            ResponseSchema(name="domain",      description="Return the domain name"),
            ResponseSchema(name="elaboration", description="Detailed technical elaboration (≤ 200 words)"),
            ResponseSchema(name="references",  description="List of 2-4 reference titles (no URLs needed)")
        ]
        self.parser = StructuredOutputParser.from_response_schemas(self.schemas)

        self.prompt = PromptTemplate(
            template=(
                "You are a senior researcher in {domain}.\n"
                "Given the following cross-disciplinary hypothesis JSON, produce an in-depth "
                "domain-specific elaboration and cite 2-4 key references.\n\n"
                "HYPOTHESIS JSON:\n{onto_json}\n\n"
                "{format_instructions}"
            ),
            input_variables=["onto_json"],
            partial_variables={
                "domain": self.domain,
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def run(self, onto_json: Dict[str, Any]) -> Dict[str, Any]:
        prompt_msg = self.prompt.format(onto_json=json.dumps(onto_json, indent=2))
        llm_out = LLM.invoke(prompt_msg).content
        return self.parser.parse(llm_out)

class CriticAgent:
    """ novelty: float, feasbility: float, comments: str """
    def __init__(self):
        self.schemas = [
            ResponseSchema(name="novelty",     description="Novelty score 0-1"),
            ResponseSchema(name="feasibility", description="Feasibility score 0-1"),
            ResponseSchema(name="comments",    description="Critical review paragraph")
        ]
        self.parser = StructuredOutputParser.from_response_schemas(self.schemas)
        self.prompt = PromptTemplate(
            template=(
                "You are a meticulous scientific reviewer.\n"
                "Evaluate the following cross-domain elaborations.\n\n"
                "ELABORATIONS:\n{domain_json}\n\n"
                "Rate novelty (0-1) and feasibility (0-1). Explain in comments.\n"
                "{format_instructions}"
            ),
            input_variables=["domain_json"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            }
        )

    def run(self, domain_json_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_msg = self.prompt.format(domain_json=json.dumps(domain_json_list, indent=2))
        llm_out = LLM.invoke(prompt_msg).content
        return self.parser.parse(llm_out)


def run_pipeline(path: List[Dict[str, str]]) -> Dict[str, Any]:
    onto = OntologistAgent().run(path)
    bio = DomainExpertAgent("Biomaterials").run(onto)
    robo = DomainExpertAgent("Robotics").run(onto)
    critic = CriticAgent().run([bio, robo])
    return {"ontologist": onto, "domains": [bio, robo], "critic": critic}


if __name__ == "__main__":
    sample_path = [
        {"head": "spider silk", "relation": "related_to", "tail": "protein structure"},
        {"head": "protein structure", "relation": "influences", "tail": "mechanical properties"},
        {"head": "mechanical properties", "relation": "applied_in", "tail": "soft robotics"}
    ]

    report = run_pipeline(sample_path)
    print("SUCCESS", json.dumps(report, indent=2))


