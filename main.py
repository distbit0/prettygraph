from typing import Generic, TypeVar

import marvin
from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel

app = Flask(__name__)

T = TypeVar("T")


class Item(BaseModel, Generic[T]):
    data: T


class Node(BaseModel):
    id: str
    label: str


class Edge(BaseModel):
    source: str
    target: str
    label: str


class Graph(BaseModel):
    """
    MUST use a format where we can jsonify in python and feed directly
    into cy.add(data); to display a graph on the front-end.
    """

    nodes: list[Item[Node]]
    edges: list[Item[Edge]]


@marvin.fn(model_kwargs={"model": "gpt-4-turbo-preview"})
def make_graph(text: str) -> Graph:
    """You are an AI expert specializing in knowledge graph creation with the goal of
    capturing relationships based on a given input or request. Based on the user input
    in various forms such as paragraph, email, text files, and more. Your task is to create
    a knowledge graph based on the input. Nodes must have a label parameter, where the label
    is a direct word or phrase from the input. Edges must also have a label parameter, where
    the label is a direct word or phrase from the input.
    """


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/update_graph", methods=["POST"])
def update_graph():
    text = request.json.get("text", "")
    graph_data = make_graph(text)
    return jsonify(graph_data.model_dump())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
