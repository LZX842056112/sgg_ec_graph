from langchain_core.messages import ToolMessage
from langchain_neo4j import Neo4jGraph
from agent import get_agent
from configuration.config import NEO4J_CONFIG, AGENT_STREAM_OUTPUT
import logging

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG["uri"],
            username=NEO4J_CONFIG["auth"][0],
            password=NEO4J_CONFIG["auth"][1],
        )
        self.agent = get_agent(self.graph.schema)

    def chat(self, user_query: str, session_id: str):
        agent_config = {"configurable": {"thread_id": session_id}}
        if AGENT_STREAM_OUTPUT:
            for msg, _ in self.agent.stream(
                    {"messages": [("user", user_query)]},
                    config=agent_config,
                    stream_mode="messages"
            ):
                if isinstance(msg, ToolMessage) or not msg.content:
                    continue
                yield msg.content
        else:
            result = self.agent.invoke(
                {"messages": [("user", user_query)]},
                config=agent_config
            )
            yield result["messages"][-1].content
