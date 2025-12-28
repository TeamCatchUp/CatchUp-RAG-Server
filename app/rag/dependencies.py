from functools import lru_cache
from fastapi import Depends
from app.rag.service.search import SearchService
from app.rag.repository.base import VectorStoreRepository
from app.rag.repository.meili import LangChainMeiliRepository

@lru_cache # 싱글톤
def get_vector_repository() -> VectorStoreRepository:
    return LangChainMeiliRepository()


# 검색 서비스 의존성 주입
def get_search_service(
        repo: VectorStoreRepository = Depends(get_vector_repository)
) -> SearchService:
    return SearchService(repo)