import logging
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from agent.schema import CypherCheckerResponse
from agent.prompts import cypher_checker_prompt
from entity_alignment import EntityAlignment
from configuration.config import API_KEY, NEO4J_CONFIG
from langchain_neo4j import Neo4jGraph
import neo4j

logger = logging.getLogger(__name__)

# 初始化 Neo4j 驱动和 Schema
neo4j_driver = neo4j.GraphDatabase.driver(NEO4J_CONFIG['uri'], auth=NEO4J_CONFIG['auth'])
neo4j_graph = Neo4jGraph(url=NEO4J_CONFIG['uri'], username=NEO4J_CONFIG['auth'][0], password=NEO4J_CONFIG['auth'][1])
neo4j_schema = neo4j_graph.schema

cypher_checker_llm = ChatDeepSeek(model="deepseek-chat", api_key=API_KEY).with_structured_output(CypherCheckerResponse)
ea = EntityAlignment()


def entity_alignment_tool(entitys_to_alignment: list):
    """
    实体对齐工具
    输入: [{"entity": "Apple", "label": "Trademark"}, ...]
    输出: 对齐后的列表
    """
    logger.info(f"实体对齐工具调用: {entitys_to_alignment}")
    for item in entitys_to_alignment:
        aligned = ea(item["entity"], item["label"])
        if aligned:
            item["entity"] = aligned
    return entitys_to_alignment


def check_syntax_error(cypher: str):
    """Cypher语法校验工具"""
    logger.info(f"校验Cypher: {cypher}")
    prompt = PromptTemplate.from_template(cypher_checker_prompt)
    chain = prompt | cypher_checker_llm
    result = chain.invoke({"neo4j_schema": neo4j_schema, "cypher": cypher})
    logger.info(f"校验结果: {result}")
    return result


def neo4j_query_tool(cypher: str, params: dict = None):
    """执行Cypher查询工具"""
    logger.info(f"执行查询: {cypher}, 参数: {params}")
    if params is None:
        params = {}
    result = neo4j_driver.execute_query(cypher, parameters_=params)
    return result.records
