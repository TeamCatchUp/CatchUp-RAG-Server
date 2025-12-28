from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from app.core.config import settings

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
        세션 ID를 기준으로 LangChain 대화 히스토리 객체를 반환한다.
        
        주어진 세션 ID로 Redis에 저장된 대화 내역이 있다면 로드하고, 
        없다면 새로운 대화 기록을 시작하도록 초기화한다.
        
        데이터는 TTL 설정에 따라 24시간 동안 유지된다.
        
        Args:
            session_id (str): 세션 ID
            
        Returns:
            BaseChatMessageHistory: Redis와 연동된 대화 메세지 기록 객체
        
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=86400
    )