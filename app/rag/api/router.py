from fastapi import APIRouter, Depends
from app.rag.dto import *
from app.rag.service.search import SearchService
from app.rag.dependencies import get_search_service

router = APIRouter()

@router.post("/api/chat/")
async def chat_response(
    request: QueryRequest,
    service: SearchService = Depends(get_search_service)
) -> QueryResponse:
    """
        사용자 질의에 대해 Hybrid Search & RAG 기반 LLM 응답을 반환한다.
        
        입력된 query를 기반으로 벡터 검색 및 LLM 호출을 수행하여 최종 응답을 반환한다.
        세션 ID는 대화 맥락 추적에 사용된다.
        
        Args:
            request (QueryRequest):
                사용자 질의 요청 객체.
                - query (str): 사용자의 자연어 질문
                - role (str): 사용자 역할
                - session_id (str): 대화 세션 식별자
        
        Returns:
            response (QueryResponse):
                LLM이 생성한 최종 응답 텍스트 및 출처
                - answer (str): LLM 최종 응답
                - source (str): 응답 출처 (없으면 Unknown)
    """
    
    response: str = await service.get_search_result(
        query=request.query,
        role=request.role,
        session_id=request.session_id,
        k=5
    )
    
    return response