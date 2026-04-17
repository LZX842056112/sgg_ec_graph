from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from agent.tools_def import entity_alignment_tool, check_syntax_error, neo4j_query_tool
from agent.prompts import major_agent_system_prompt_template
from configuration.config import API_KEY


def get_agent(neo4j_schema: str):
    llm = ChatDeepSeek(model="deepseek-chat", api_key=API_KEY)
    tools = [entity_alignment_tool, check_syntax_error, neo4j_query_tool]
    system_prompt = major_agent_system_prompt_template.format(neo4j_schema=neo4j_schema)

    # 兼容不同版本的 langgraph
    try:
        # 新版本参数名：prompt
        agent = create_react_agent(llm, tools, prompt=system_prompt)
    except TypeError:
        try:
            # 旧版本参数名：state_modifier
            agent = create_react_agent(llm, tools, state_modifier=system_prompt)
        except TypeError:
            # 极新版本可能使用 system_prompt
            agent = create_react_agent(llm, tools, system_prompt=system_prompt)
    return agent
