import os
import json
import re
from typing import List, Dict, Any, Optional
import pypdf
import chromadb
from chromadb.utils import embedding_functions

# Model configuration: comprehensive fallback list covering all stable tiers
DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-pro"
]

class NCERTKnowledgeBase:
    """Manages PDF document ingestion, chunking, and vector retrieval."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "ncert_history"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Lightweight, high-performance local embedding model (runs free on Colab CPU/GPU)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extracts text page-by-page from a PDF file."""
        reader = pypdf.PdfReader(pdf_path)
        pages_data = []
        filename = os.path.basename(pdf_path)
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "text": text.strip(),
                    "page": idx + 1,
                    "source": filename
                })
        return pages_data

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
        """Simple, clean sliding-window chunking."""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += (chunk_size - overlap)
            if i >= len(words) - overlap:
                break
        return chunks if chunks else [text]

    def index_pdf(self, pdf_path: str):
        """Extracts, chunks, and indexes a PDF into ChromaDB."""
        pages = self.extract_text_from_pdf(pdf_path)
        documents = []
        metadatas = []
        ids = []

        for p in pages:
            chunks = self.chunk_text(p["text"])
            for c_idx, chunk in enumerate(chunks):
                doc_id = f"{p['source']}_p{p['page']}_c{c_idx}"
                documents.append(chunk)
                metadatas.append({
                    "source": p["source"],
                    "page": p["page"],
                    "chunk_id": c_idx
                })
                ids.append(doc_id)

        if documents:
            # Batch upsert into Chroma
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        return len(documents)

    def index_directory(self, data_dir: str) -> Dict[str, int]:
        """Indexes all PDFs in the given directory."""
        results = {}
        if not os.path.exists(data_dir):
            return results

        for fname in os.listdir(data_dir):
            if fname.lower().endswith(".pdf"):
                full_path = os.path.join(data_dir, fname)
                count = self.index_pdf(full_path)
                results[fname] = count
        return results

    def retrieve_evidence(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieves top-k relevant evidence chunks for a question."""
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = self.collection.query(
            query_texts=[query],
            n_results=actual_k
        )
        
        evidence_list = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            for doc, meta, dist in zip(docs, metas, dists):
                evidence_list.append({
                    "content": doc,
                    "source": meta.get("source", "NCERT Book"),
                    "page": meta.get("page", 1),
                    "score": round(1.0 - (dist if dist is not None else 0.5), 3)
                })
        return evidence_list


class HallucinationEvaluator:
    """Evaluates AI answers against retrieved RAG evidence according to the methodology."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.client = None
        if self.api_key:
            self._init_gemini_client()

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()
        self._init_gemini_client()

    def _init_gemini_client(self):
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            # Fallback to google.generativeai if available
            try:
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self.client = "classic"
            except Exception as e2:
                self.client = None
                print(f"Warning: Failed to initialize Gemini client: {e2}")

    def _call_gemini(self, prompt: str) -> str:
        import time
        if not self.client:
            raise RuntimeError("Gemini Client not initialized.")
        last_err = None
        for m in FALLBACK_MODELS:
            for attempt in range(2):
                try:
                    if self.client == "classic":
                        import google.generativeai as gai
                        model_obj = gai.GenerativeModel(m)
                        return model_obj.generate_content(prompt).text.strip()
                    else:
                        resp = self.client.models.generate_content(model=m, contents=prompt)
                        return resp.text.strip()
                except Exception as e:
                    last_err = e
                    err_str = str(e).upper()
                    if any(k in err_str for k in ['503', 'UNAVAILABLE', 'HIGH DEMAND', '429', 'RESOURCE_EXHAUSTED', '404', 'NOT_FOUND']):
                        time.sleep(1.5)
                        continue
                    break
        raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")

    def generate_candidate_answer(self, question: str) -> str:
        """Simulates or generates an answer directly from an LLM without RAG (to test hallucination)."""
        if not self.client:
            return "Please configure your GEMINI_API_KEY to generate answers."

        prompt = f"""You are answering a student's question based on Indian NCERT History & Civics:
Question: {question}

Answer the question clearly and directly in 2-4 sentences."""
        try:
            return self._call_gemini(prompt)
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def evaluate(self, question: str, candidate_answer: str, retrieved_evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compares evidence with the candidate answer and produces:
        - Verdict (e.g. Grounded / Hallucinated / Partially Hallucinated)
        - Confidence (e.g. 95% / High)
        - Evidence (Exact quotes/excerpts from NCERT)
        - Status (e.g. Verified / Contradicted / Unsupported)
        - Explanation (Step-by-step reasoning)
        """
        if not retrieved_evidences:
            return {
                "verdict": "UNVERIFIABLE",
                "confidence": "50%",
                "status": "No Evidence Found",
                "evidence": "No relevant text could be retrieved from the NCERT knowledge base.",
                "explanation": "The knowledge base did not return relevant passages for this question. Unable to cross-verify factual statements."
            }

        # Format evidence context
        evidence_text = "\n\n".join([
            f"[Source: {e['source']}, Page: {e['page']}]\n{e['content']}"
            for e in retrieved_evidences
        ])

        eval_prompt = f"""You are an expert AI Hallucination Evaluator for educational and historical content.
Your task is to perform an objective verification comparing an AI-generated answer against the ground-truth textbook evidence retrieved from NCERT textbooks.

[QUESTION]:
{question}

[AI GENERATED ANSWER]:
{candidate_answer}

[RETRIEVED NCERT TEXTBOOK EVIDENCE]:
{evidence_text}

INSTRUCTIONS:
1. Examine each factual claim in the [AI GENERATED ANSWER] against the [RETRIEVED NCERT TEXTBOOK EVIDENCE].
2. Identify:
   - Contradicted facts (claims that directly conflict with the evidence)
   - Fabricated/unsupported facts (claims not mentioned in or substantiated by the evidence)
   - Grounded facts (claims verified by the evidence)
3. Determine:
   - VERDICT: Must be one of ["FAITHFUL / GROUNDED", "HALLUCINATED", "PARTIALLY HALLUCINATED"].
   - CONFIDENCE: Confidence percentage (e.g. "95%", "85%").
   - EVIDENCE: The specific key quote(s) and page numbers from the retrieved evidence that prove or disprove the answer.
   - STATUS: One of ["Verified / Supported", "Contradicted", "Unsupported / Fabricated", "Partially Supported"].
   - EXPLANATION: Comprehensive step-by-step breakdown explaining why the answer is faithful or where and how it hallucinates.

Return strictly a valid JSON object with EXACTLY this structure (no markdown wrappers like ```json, just raw JSON):
{{
  "verdict": "VERDICT_VALUE",
  "confidence": "CONFIDENCE_VALUE",
  "status": "STATUS_VALUE",
  "evidence": "EXACT_QUOTES_WITH_CITATION",
  "explanation": "STEP_BY_STEP_REASONING"
}}"""

        if not self.client:
            # Fallback simulation if no API key is provided
            return {
                "verdict": "API KEY MISSING",
                "confidence": "0%",
                "status": "Pending Key",
                "evidence": evidence_text[:300] + "...",
                "explanation": "Please enter your Gemini API Key in the settings or environment to enable real-time LLM evaluation."
            }

        try:
            raw_text = self._call_gemini(eval_prompt)

            # Clean JSON formatting
            cleaned_json = re.sub(r"^```json\s*", "", raw_text, flags=re.MULTILINE)
            cleaned_json = re.sub(r"^```\s*", "", cleaned_json, flags=re.MULTILINE)
            cleaned_json = cleaned_json.strip()

            data = json.loads(cleaned_json)
            return {
                "verdict": data.get("verdict", "UNKNOWN"),
                "confidence": str(data.get("confidence", "N/A")),
                "status": data.get("status", "Analyzed"),
                "evidence": data.get("evidence", evidence_text[:300]),
                "explanation": data.get("explanation", raw_text)
            }
        except Exception as e:
            return {
                "verdict": "EVALUATION ERROR",
                "confidence": "0%",
                "status": "Failed",
                "evidence": evidence_text[:300] + "...",
                "explanation": f"Evaluation parsing error: {str(e)}\nRaw Response: {raw_text if 'raw_text' in locals() else ''}"
            }
