from app.rag.repository.base import VectorStoreRepository
from app.rag.service.llm import LlmService
from app.rag.dto import QueryResponse

class SearchService:
    def __init__(self, repository: VectorStoreRepository):
        self.repository = repository
        self.llm_service = LlmService()
    
    async def get_search_result(self, query: str, role: str, session_id: str, k: int = 3) -> QueryResponse:
        """
            사용자 요청에 대해 검색을 수행하고 LLM 응답을 생성한다.
            
            Args:
                - query (str): 사용자의 자연어 질문
                - role (str): 사용자 역할
                - session_id (str): 대화 세션 식별자
                - k (int): 상위 k개의 Document (default: 3)
                
            Returns:
                response (QueryResponse):
                    LLM이 생성한 최종 응답 텍스트 및 출처
                    - answer (str): LLM 최종 응답
                    - source (str): 응답 출처 (없으면 Unknown)            
        """
        docs = self.repository.retrieve(query, k)
        return await self.llm_service.generate_answer(query, session_id, role, docs)