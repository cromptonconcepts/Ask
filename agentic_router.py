"""
Multi-Agent Agentic RAG System using LangGraph
===============================================

Three specialized agents work together:
1. Router Agent     - Analyzes question, routes to AGTTM/QGTTM/Both
2. Researcher Agent - Generates optimized search queries, retrieves chunks
3. Engineer Agent   - Synthesizes final answer from retrieved context

Each agent has a specific purpose and hands off to the next in the pipeline.
"""

import logging
import os
from typing import TypedDict, Optional, List, Dict, Any, cast
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
import json
import re

logger = logging.getLogger("agentic_router")
logger.setLevel(logging.INFO)

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class AgenticRAGState(TypedDict):
    """The shared state passed through the agent workflow."""
    question: str
    selected_document: Optional[str]
    session_id: Optional[str]
    conversation_context: str
    search_hints: List[str]
    
    # Router output
    route_decision: Optional[str]  # "AGTTM", "QGTTM", or "BOTH"
    route_confidence: Optional[float]
    route_reasoning: Optional[str]
    
    # Researcher output
    search_queries: List[str]
    retrieved_documents: List[Document]
    search_metadata: Dict[str, Any]
    
    # Engineer output
    final_answer: str
    answer_confidence: Optional[float]
    sources_used: List[Dict[str, Any]]


class AgenticRAGUpdate(TypedDict, total=False):
    """Partial state updates produced by each agent node."""
    route_decision: Optional[str]
    route_confidence: Optional[float]
    route_reasoning: Optional[str]
    search_hints: List[str]
    search_queries: List[str]
    retrieved_documents: List[Document]
    search_metadata: Dict[str, Any]
    final_answer: str
    answer_confidence: Optional[float]
    sources_used: List[Dict[str, Any]]


# ============================================================================
# ROUTER AGENT - Intelligently decides which documents to search
# ============================================================================

class RouterAgent:
    """
    Analyzes incoming questions to determine:
    - Which standard(s) are relevant (AGTTM, QGTTM, or BOTH)
    - What specific aspects to focus on
    - Whether document-type constraints exist
    """
    
    def __init__(self, llm: Any):
        self.llm = llm

    def invoke(self, state: AgenticRAGState) -> AgenticRAGUpdate:
        """Route using fast keyword matching — no LLM call needed."""
        question_lower = state['question'].lower()
        selected = (state.get('selected_document') or '').lower()
        combined = question_lower + ' ' + selected

        qgttm_keywords = ['queensland', 'qgttm', 'qld']
        agttm_keywords = ['federal', 'national', 'agttm', 'austroads', 'australian guide']

        is_qgttm = any(k in combined for k in qgttm_keywords)
        is_agttm = any(k in combined for k in agttm_keywords)

        if is_qgttm and not is_agttm:
            route, confidence, reasoning = "QGTTM", 0.9, "Queensland-specific terms detected"
        elif is_agttm and not is_qgttm:
            route, confidence, reasoning = "AGTTM", 0.9, "National/federal terms detected"
        else:
            route, confidence, reasoning = "BOTH", 0.7, "General question — searching both standards"

        return {
            "route_decision": route,
            "route_confidence": confidence,
            "route_reasoning": reasoning,
            "search_hints": []
        }


# ============================================================================
# RESEARCHER AGENT - Optimizes search queries and retrieves documents
# ============================================================================

class ResearcherAgent:
    """
    Transforms questions into optimal search queries:
    - Decomposes complex questions into multiple search angles
    - Filters based on document type (AGTTM/QGTTM)
    - Retrieves chunks using hybrid retrieval
    """
    
    def __init__(self, llm: Any, vectorstore, reranker):
        self.llm = llm
        self.vectorstore = vectorstore
        self.reranker = reranker
    
    def generate_search_queries(self, state: AgenticRAGState, search_hints: List[str]) -> List[str]:
        """Return the original question directly — fast, no LLM call needed."""
        return [state['question']]
    
    def retrieve_documents(self, queries: List[str], selected_document: Optional[str] = None) -> List[Document]:
        """Retrieve documents using hybrid retrieval (MMR + reranking)."""
        all_docs = []
        seen_ids = set()
        
        for query in queries:
            # Stage 1: Vector similarity search (MMR-like)
            if self.vectorstore:
                docs = self.vectorstore.similarity_search(query, k=10)
                
                # Stage 2: Cross-encoder re-ranking
                if self.reranker and docs:
                    from sentence_transformers import CrossEncoder
                    scores = self.reranker.predict([[query, doc.page_content] for doc in docs])
                    scored_docs = list(zip(docs, scores))
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    docs = [doc for doc, score in scored_docs[:6]]
                
                # Filter for selected document if specified
                if selected_document:
                    selected_norm = selected_document.lower().replace("\\", "/")
                    docs = [d for d in docs if selected_norm in d.metadata.get("source", "").lower().replace("\\", "/")]
                
                # Deduplicate
                for doc in docs:
                    doc_id = id(doc)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
        
        return all_docs[:20]  # Return top 20 most relevant docs
    
    def invoke(self, state: AgenticRAGState, search_hints: List[str]) -> AgenticRAGUpdate:
        """Generate queries and retrieve documents."""
        queries = self.generate_search_queries(state, search_hints)
        documents = self.retrieve_documents(queries, state.get('selected_document'))
        
        return {
            "search_queries": queries,
            "retrieved_documents": documents,
            "search_metadata": {
                "num_queries": len(queries),
                "num_docs_retrieved": len(documents),
                "route": state.get('route_decision')
            }
        }


# ============================================================================
# ENGINEER AGENT - Synthesizes final answer from context
# ============================================================================

class EngineerAgent:
    """
    Takes retrieved context and synthesizes the final answer:
    - Formats context intelligently
    - Considers conversation history
    - Generates comprehensive response
    """
    
    def __init__(self, llm: Any):
        self.llm = llm
        
        self.synthesis_prompt = ChatPromptTemplate.from_template("""
You are a traffic management engineer. Answer the question using the provided context.

Question: {question}

Context from Traffic Management Standards:
{context}

Conversation History (if any):
{conversation}

Provide a clear, practical answer. Reference specific standards (AGTTM/QGTTM) when applicable.
If comparing standards, highlight key differences.

Answer:
""")
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into readable context."""
        if not documents:
            return "No relevant documents retrieved."
        
        formatted = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown").split("/")[-1]
            page = doc.metadata.get("page", "?")
            formatted.append(f"[{i}] {source} (p. {page}):\n{doc.page_content[:500]}...\n")
        
        return "\n".join(formatted)
    
    def invoke(self, state: AgenticRAGState) -> AgenticRAGUpdate:
        """Synthesize the final answer."""
        context = self.format_context(state['retrieved_documents'])
        
        filled_prompt = self.synthesis_prompt.format_messages(
            question=state['question'],
            context=context,
            conversation=state.get('conversation_context', "None")
        )
        
        try:
            response = self.llm.invoke(filled_prompt)
            if hasattr(response, 'content'):
                response = response.content
            
            # Extract sources
            sources = []
            for doc in state['retrieved_documents']:
                source_entry = {
                    "title": doc.metadata.get("source", "").split("/")[-1],
                    "path": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page"),
                }
                if source_entry not in sources:
                    sources.append(source_entry)
            
            return {
                "final_answer": str(response),
                "answer_confidence": 0.85,
                "sources_used": sources
            }
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return {
                "final_answer": "Error generating answer. Please try again.",
                "answer_confidence": 0.0,
                "sources_used": []
            }

    def stream(self, state: AgenticRAGState):
        """Stream the synthesis response token by token."""
        context = self.format_context(state['retrieved_documents'])
        filled_prompt = self.synthesis_prompt.format_messages(
            question=state['question'],
            context=context,
            conversation=state.get('conversation_context', "None")
        )
        try:
            for chunk in self.llm.stream(filled_prompt):
                yield chunk
        except Exception as e:
            logger.error(f"Synthesis stream error: {e}")
            yield f"\n\nError generating answer: {str(e)}"


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

class AgenticRAGSystem:
    """
    Orchestrates the multi-agent workflow using LangGraph.
    
    Flow:
    Original Question → Router → Researcher → Engineer → Final Answer
    """
    
    def __init__(self, llm: Any, vectorstore, reranker):
        self.llm = llm
        self.vectorstore = vectorstore
        self.reranker = reranker
        
        # Initialize agents
        self.router = RouterAgent(llm)
        self.researcher = ResearcherAgent(llm, vectorstore, reranker)
        self.engineer = EngineerAgent(llm)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgenticRAGState)
        
        # Add nodes for each agent
        workflow.add_node("router", self._route_node)
        workflow.add_node("researcher", self._researcher_node)
        workflow.add_node("engineer", self._engineer_node)
        
        # Define edges
        workflow.set_entry_point("router")
        workflow.add_edge("router", "researcher")
        workflow.add_edge("researcher", "engineer")
        workflow.add_edge("engineer", END)
        
        return workflow.compile()

    def _merge_state(self, state: AgenticRAGState, updates: AgenticRAGUpdate) -> AgenticRAGState:
        """Merge a partial update into the typed workflow state."""
        cast(Dict[str, Any], state).update(updates)
        return state
    
    def _route_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Router node - determines document type to search."""
        route_result = self.router.invoke(state)
        state = self._merge_state(state, route_result)
        logger.info(
            "Router: %s (confidence: %.2f)",
            route_result.get("route_decision", "BOTH"),
            float(route_result.get("route_confidence", 0.0) or 0.0)
        )
        return state
    
    def _researcher_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Researcher node - generates queries and retrieves docs."""
        search_hints = state.get('search_hints', [])
        researcher_result = self.researcher.invoke(state, search_hints)
        state = self._merge_state(state, researcher_result)
        logger.info(
            "Researcher: Generated %d queries, retrieved %d docs",
            len(researcher_result.get("search_queries", [])),
            len(researcher_result.get("retrieved_documents", []))
        )
        return state
    
    def _engineer_node(self, state: AgenticRAGState) -> AgenticRAGState:
        """Engineer node - synthesizes the final answer."""
        engineer_result = self.engineer.invoke(state)
        state = self._merge_state(state, engineer_result)
        logger.info(
            "Engineer: Answer generated (confidence: %.2f)",
            float(engineer_result.get("answer_confidence", 0.0) or 0.0)
        )
        return state
    
    def invoke(
        self,
        question: str,
        selected_document: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_context: str = ""
    ) -> Dict[str, Any]:
        """
        Run the multi-agent system on a question.
        
        Returns:
            {
                "answer": str,
                "sources": List[Dict],
                "routing": {route, confidence, reasoning},
                "search_queries": List[str],
                "agent_details": {...}
            }
        """
        initial_state = AgenticRAGState(
            question=question,
            selected_document=selected_document,
            session_id=session_id,
            conversation_context=conversation_context,
            search_hints=[],
            route_decision=None,
            route_confidence=None,
            route_reasoning=None,
            search_queries=[],
            retrieved_documents=[],
            search_metadata={},
            final_answer="",
            answer_confidence=None,
            sources_used=[]
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Format output
        return {
            "answer": final_state.get("final_answer"),
            "sources": final_state.get("sources_used", []),
            "routing": {
                "decision": final_state.get("route_decision"),
                "confidence": final_state.get("route_confidence"),
                "reasoning": final_state.get("route_reasoning")
            },
            "search_queries": final_state.get("search_queries", []),
            "agent_details": {
                "router_confidence": final_state.get("route_confidence"),
                "answer_confidence": final_state.get("answer_confidence"),
                "num_documents_retrieved": len(final_state.get("retrieved_documents", [])),
                "num_search_queries": len(final_state.get("search_queries", []))
            }
        }

    def stream(
        self,
        question: str,
        selected_document: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_context: str = ""
    ):
        """Run router+researcher synchronously, then stream engineer tokens.
        Yields dicts: first {'type':'metadata',...}, then {'type':'token','content':str}, then {'type':'done'}.
        """
        initial_state = AgenticRAGState(
            question=question,
            selected_document=selected_document,
            session_id=session_id,
            conversation_context=conversation_context,
            search_hints=[],
            route_decision=None, route_confidence=None, route_reasoning=None,
            search_queries=[], retrieved_documents=[], search_metadata={},
            final_answer="", answer_confidence=None, sources_used=[]
        )
        # Fast phases (keyword routing + vector retrieval, no LLM)
        state = self._route_node(initial_state)
        state = self._researcher_node(state)

        # Build sources list
        sources: List[Dict[str, Any]] = []
        seen_paths: set = set()
        for doc in state['retrieved_documents']:
            path = doc.metadata.get('source', '')
            if path not in seen_paths:
                seen_paths.add(path)
                sources.append({
                    'title': os.path.basename(path),
                    'path': path,
                    'page': doc.metadata.get('page')
                })

        yield {
            'type': 'metadata',
            'routing': {
                'mode': 'agentic',
                'router_decision': state['route_decision'],
                'router_confidence': state['route_confidence'],
                'router_reasoning': state['route_reasoning'],
                'search_queries': state['search_queries'],
                'num_documents_retrieved': len(state['retrieved_documents'])
            },
            'sources': sources,
            'session_id': session_id
        }

        # Stream engineer synthesis token by token
        for chunk in self.engineer.stream(state):
            yield {'type': 'token', 'content': chunk}

        yield {'type': 'done'}
