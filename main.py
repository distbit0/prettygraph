from typing import Generic, TypeVar

import logfire
import marvin
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from marvin.beta.retries import retry_fn_on_validation_error
from marvin.client import AsyncMarvinClient
from openai import AsyncClient
from pydantic import BaseModel, Field, model_validator

app = FastAPI()
client = AsyncClient()
logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record="all"))
logfire.instrument_fastapi(app)
logfire.instrument_openai(client)

templates = Jinja2Templates(directory="templates")

T = TypeVar("T")


class Item(BaseModel, Generic[T]):
    data: T


class Node(BaseModel):
    id: str = Field(..., description="Unique, human-readable ID for the node")
    label: str = Field(..., description="Unaltered word or phrase from the input")


class Edge(BaseModel):
    """both source and target must be the ID of an EXISTING node"""

    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    label: str = Field(..., description="Direct word or phrase from the input")


class Graph(BaseModel):
    """Represents a knowledge graph based on the input.
    Format must be compatible with cy.add(data) for displaying the graph on the frontend
    """

    nodes: list[Item[Node]] = Field(description="List of nodes in the knowledge graph")
    edges: list[Item[Edge]] = Field(description="List of edges in the knowledge graph")

    @model_validator(mode="after")
    def ensure_all_sources_and_targets_exist(self):
        """surface errors with non-existing node so the LLM can try again"""
        errors = []
        node_ids = {node.data.id for node in self.nodes}
        for edge in self.edges:
            if edge.data.source not in node_ids:
                errors.append(f"Source node {edge.data.source} not found in nodes")
            if edge.data.target not in node_ids:
                errors.append(f"Target node {edge.data.target} not found in nodes")
        if errors:
            raise ValueError(errors)
        return self


@retry_fn_on_validation_error  # reask + content of validation error(s)
@marvin.fn(
    model_kwargs={"model": "gpt-4-turbo-preview", "temperature": 0.5},
    client=AsyncMarvinClient(client=client),
)
def make_graph(text: str) -> Graph:  # noqa
    """You are an AI expert specializing in knowledge graph creation with the goal of
    capturing relationships based on a given input or request. Based on the user input
    in various forms such as paragraph, email, text files, and more. Your task is to
    create a knowledge graph based on the input.
    """


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/update_graph")
async def update_graph(request: Request):
    data = await request.json()
    text = data.get("text", "")
    graph_data = make_graph(text)
    return graph_data.model_dump()
