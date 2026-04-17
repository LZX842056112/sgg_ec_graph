from pydantic import BaseModel, Field
from typing import Dict


class CypherCheckerResponse(BaseModel):
    is_legal: bool = Field(description="Cypher是否合法")
    error_msg: str = Field(description="错误信息，合法时为空")
    solve_method: str = Field(description="修正建议，合法时为空")
    original_cypher: str = Field(description="原始Cypher语句")


class CheckSyntaxError(BaseModel):
    cypher: str = Field(description="待校验的Cypher语句")


class Neo4jQueryParams(BaseModel):
    cypher: str = Field(description="Cypher查询语句")
    params: Dict[str, str] = Field(default_factory=dict, description="参数字典")


class EntityAlignmentList(BaseModel):
    entitys_to_alignment: list[Dict[str, str]] = Field(description="待对齐实体列表")
