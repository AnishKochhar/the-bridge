# LLM-driven agents

""" .env contains OPENAI_API_KEY  """

from  __future__ import annotations

import json, tiktoken
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI                                     # GPT-4 wrapper
from langchain_core.prompts import PromptTemplate                           # Prompt building
from langchain.output_parsers import StructuredOutputParser, ResponseSchema # JSON parsing
from openai import OpenAIError
from openai.lib._parsing._completions import LengthFinishReasonError

from arxiv_search import hybrid_search

MAX_CONTEXT = 16384

load_dotenv()

def num_tokens(text: str, model: str = "gpt-4o") -> int:
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text))

# Ontologist
class OntologistAgent:
    """ hypothesis: str, mechanism: str, assumptions: [str] """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3,
            model_kwargs={"response_format": {"type":"json_object"}})
        self.schemas = [
            ResponseSchema(name="hypothesis",  description="One concise sentence"),
            ResponseSchema(name="mechanism",   description="1-2 sentences, linking each edge"),
            ResponseSchema(name="assumptions", description="list of explicit assumptions", type="array")
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

    def run(self, path_str: str) -> Dict[str, Any]:
        prompt_msg = self.prompt.format(path_str=path_str)
        llm_out = self.llm.invoke(prompt_msg).content
        try:
            return self.parser.parse(llm_out)
        except Exception:
            print(" !! Parsing Failed !!")
            print("LLM Output:\n", llm_out)
            return {"hypothesis": llm_out, "mechanism": "", "assumptions": ""}


class DomainExpertAgent:
    """ domain: str, elaboration: str, references: [str] """
    def __init__(self, domain: str):
        self.domain = domain
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3,
            model_kwargs={"response_format": {"type":"json_object"}})

        self.schemas = [
            ResponseSchema(name="domain",      description="Return the same domain name"),
            ResponseSchema(name="elaboration", description="Detailed technical elaboration (200 words)"),
            ResponseSchema(name="references",  description="List of 2-4 reference titles", type="array")
        ]
        self.parser = StructuredOutputParser.from_response_schemas(self.schemas)

        self.prompt = PromptTemplate(
            template=(
                "You are a leading researcher in {domain}.\n"
                "You are given a hypothesis extracted from a knowledge graph, along with a set of domain-specific papers.\n\n"
                "Hypothesis JSON:\n{hyp_json}\n\n"
                "Context documents:\n{docs}\n\n"
                "Write a technically rigorous elaboration of the hypothesis, supported by insights from the documents above. "
                "Cite 2-4 relevant papers by their title.\n\n"
                "{format_instructions}"
            ),
            input_variables=["hyp_json", "docs"],
            partial_variables={
                "domain": self.domain,
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def run(self, onto_json: Dict[str, Any]) -> Dict[str, Any]:
        # Retrieve the docs
        query = onto_json["hypothesis"]
        docs = hybrid_search(query, self.domain)
        docs_block = "\n\n".join(
            f"[{i+1}] Title: {d.metadata['title']}\nAbstract: {d.page_content}"
            for i, d in enumerate(docs)
        )

        prompt_msg = self.prompt.format(hyp_json=json.dumps(onto_json, indent=2), docs=docs_block)
        llm_out = self.llm.invoke(prompt_msg).content
        try:
            return self.parser.parse(llm_out)
        except Exception:
            print(" !! Parsing Failed !!")
            print("LLM Output:\n", llm_out)
            return {"domain": self.domain, "elaboration": llm_out, "references": []}

class BridgeAgent:
    """ bridge_idea: str, synergies: str, challenges: str, fused_references: [str] """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3,
            model_kwargs={"response_format": {"type":"json_object"}})
        self.schemas = [
            ResponseSchema(name="bridge_idea",      description="Detailed fused proposal"),
            ResponseSchema(name="synergies",        description="How domain A reinforces B (<120 words)"),
            ResponseSchema(name="challenges",       description="Integration difficulties (<120 words)"),
            ResponseSchema(name="fused_references", description="List of 3-5 paper references", type="array")
        ]
        self.parser = StructuredOutputParser.from_response_schemas(self.schemas)
        self.prompt = PromptTemplate(
            template=(
                "You are an interdisciplinary scientist tasked with integrating two domain "
                "elaborations into a single innovative idea.\n"
                "***Elaboration A (JSON)***\n{a_json}\n\n"
                "***Elaboration B (JSON)***\n{b_json}\n\n"
                "Use both perspectives, identify synergy & challenges, cite fused references.\n"
              "{format_instructions}"
            ),
            input_variables=["a_json", "b_json"],
            partial_variables={ "format_instructions": self.parser.get_format_instructions() }
        )

    def run(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        MAX_TRIES = 3
        for attempt in range(MAX_TRIES):
            try:
                prompt_msg = self.prompt.format(a_json=json.dumps(a, separators=(",", ":")), b_json=json.dumps(b, separators=(",", ":")))
                llm_out = self.llm.invoke(prompt_msg).content
                return self.parser.parse(llm_out)
            except LengthFinishReasonError:
                print(f"BridgeAgent length error attempt {attempt + 1}. Truncating input and retrying...")
                a["elaboration"] = a["elaboration"][:int(len(a["elaboration"]) * 0.7)]
                b["elaboration"] = b["elaboration"][:int(len(b["elaboration"]) * 0.7)]

            except OpenAIError as e:
                print(f"OpenAI error: {e}")

        print("Failed after retries, returning fallback")
        return {"bridge_idea": a, "synergies": "", "challenges": "", "fused_references": []}


# Update: Critic can trigger revision loops
class CriticAgent:
    """ novelty: float, feasbility: float, comments: str, recommendation: str """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3,
            model_kwargs={"response_format": {"type":"json_object"}})
        self.schemas = [
            ResponseSchema(name="novelty",        description="Novelty score 0-1"),
            ResponseSchema(name="feasibility",    description="Feasibility score 0-1"),
            ResponseSchema(name="comments",       description="Critical review paragraph"),
            ResponseSchema(name="recommendation", description="REVISE_DOMAIN | REVISE_BRIDGE | ACCEPT")
        ]
        self.parser = StructuredOutputParser.from_response_schemas(self.schemas)
        self.prompt = PromptTemplate(
            template=(
                "You are a critical interdisciplinary reviewer.\n"
                "Evaluate the following cross-domain elaborations.\n\n"
                "ELABORATIONS:\n{bridge_json}\n\n"
                "Rate novelty (0-1) and feasibility (0-1). Explain in comments.\n"
                "If novelty < 0.6 output 'REVISE_DOMAIN'. Else if references look weak output 'REVISE_BRIDGE'."
                "Else 'ACCEPT'.\n"
                "{format_instructions}"
            ),
            input_variables=["bridge_json"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            }
        )

    def run(self, bridge_json: Dict[str, Any]) -> Dict[str, Any]:
        prompt_msg = self.prompt.format(bridge_json=json.dumps(bridge_json, indent=2))
        llm_out = self.llm.invoke(prompt_msg).content
        try:
            result = self.parser.parse(llm_out)
            if result["recommendation"] not in {"REVISE_DOMAIN", "REVISE_BRIDGE", "ACCEPT"}:
                result["recommendation"] = "ACCEPT"
            return result
        except Exception:
            print(" !! Parsing Failed !!")
            print("LLM Output:\n", llm_out)
            return {"novelty": "0.1", "feasibility": "0.1", "comments": llm_out}



def run_pipeline(path: List[Dict[str, str]]) -> Dict[str, Any]:
    onto = OntologistAgent().run(path)
    bio = DomainExpertAgent("Functional Biochemistry").run(onto)
    mech = DomainExpertAgent("Bioinspired Mechanics").run(onto)
    bridge = BridgeAgent().run(bio, mech)
    critic = CriticAgent().run(bridge)
    return {"ontologist": onto, "domains": [bio, mech], "bridge": bridge, "critic": critic}


if __name__ == "__main__":
    sample_path = [
        {"head": "spider silk", "relation": "related_to", "tail": "protein structure"},
        {"head": "protein structure", "relation": "influences", "tail": "mechanical properties"},
        {"head": "mechanical properties", "relation": "applied_in", "tail": "soft robotics"}
    ]

    report = run_pipeline(sample_path)
    print("SUCCESS", json.dumps(report, indent=2))


