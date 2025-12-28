from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.documents import Document
from app.rag.service.chat_history import get_session_history
from app.rag.dto import QueryResponse

class LlmService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        # 대화 히스토리 관련 토큰 제한
        self.trimmer = trim_messages(
            max_tokens=2000,        # 토큰 제한
            strategy="last",        # 최신 부분만 남김
            token_counter=self.llm, # 토큰 계산기
            include_system=True,    # 시스템 메세지 포함
            allow_partial=False,    # 메세지 단위로 깔끔하게 자름
            start_on="human",       # 대화의 시작은 항상 사람 질문
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 {role}을 위한 사내 AI 비서입니다. 아래 [Context]를 보고 답하세요.\n\n[Context]\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        
        self.chain = (
            RunnablePassthrough.assign(
                history=lambda x: self.trimmer.invoke(x["history"]) # 대화 기록 압축
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            get_session_history,
            input_messages_key="query",
            history_messages_key="history"
        )
        
    async def generate_answer(self, query: str, role: str, session_id: str, context: list[Document]) -> QueryResponse:
        """
            사용자 맥락과 검색 결과로 얻은 맥락을 바탕으로 LLM 응답을 생성한다. 비동기적으로 동작한다.
            
            Args:
                - query (str): 사용자의 자연어 질문
                - role (str): 사용자 역할
                - session_id (str): 대화 세션 식별자
                - context (list[Document]): langchain Documnet 객체
                
            Returns:
                response (QueryResponse):
                    LLM이 생성한 최종 응답 텍스트 및 출처
                    - answer (str): LLM 최종 응답
                    - source (str): 응답 출처 (없으면 Unknown)
        """
        
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        input = {
                "role": role,
                "context": context_text,
                "query": query
            }
        
        config = {
                "configurable":
                {
                    "session_id": session_id
                }
            }
        
        answer = await self.chain_with_history.ainvoke(input, config)
        unique_source = list(set([doc.metadata.get('source', 'Unknown') for doc in context]))
        
        response = QueryResponse(answer=answer, source=unique_source)
        
        return response