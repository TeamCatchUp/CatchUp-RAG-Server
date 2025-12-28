from pydantic import BaseModel, Field
import uuid

# 사용자 쿼리 요청 양식
class QueryRequest(BaseModel):
    query: str
    role: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# 쿼리 응답 양식
class QueryResponse(BaseModel):
    answer: str
    source: list