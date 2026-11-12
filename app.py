import os
import gradio as gr
from pipeline import NCERTKnowledgeBase, HallucinationEvaluator

# Initialize KB and Evaluator
kb = NCERTKnowledgeBase(persist_directory="./chroma_db", collection_name="ncert_history")
result1 = evaluate_hallucination(q1, fake_answer, ev1)

# Auto-index data folder if exists
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
if os.path.exists(DATA_DIR):
    kb.index_directory(DATA_DIR)

# Sample test cases for Capstone presentation / demonstration
PRESETS = {
    "Select a preloaded test case...": {
        "question": "",
        "answer": ""
    },
    "[Hallucination Example 1] Battle of Plassey & Robert Clive": {
        "question": "When was the Battle of Plassey fought and who led the British forces?",
        "answer": "The Battle of Plassey was fought in 1857. The British army was led by Warren Hastings who defeated Tipu Sultan with the help of the French navy."
    },
    "[Grounded Example 1] East India Company Charter": {
        "question": "In which year did the East India Company acquire a charter from Queen Elizabeth I, and what right did it grant?",
        "answer": "In 1600, the East India Company acquired a charter from Queen Elizabeth I, granting it the sole right to trade with the East. This meant no other English trading group could compete with it."
    },
    "[Hallucination Example 2] Doctrine of Lapse Annexations": {
        "question": "What was the Doctrine of Lapse and which kingdoms were annexed under it?",
        "answer": "The Doctrine of Lapse was introduced by Lord Canning in 1840. Under this doctrine, any state without a daughter had to pay Rs 500,000 every month to the British Empire."
    },
    "[Grounded Example 2] Birsa Munda Movement": {
        "question": "Who was Birsa Munda and who were the dikus?",
        "answer": "Birsa Munda was a tribal leader born in Chottanagpur who led a movement against outsiders, known as dikus (including traders, moneylenders, and British officials), whom the tribals saw as the cause of their misery."
    },
    "[Partial Hallucination Example 3] James Mill's Periodisation": {
        "question": "How did James Mill divide Indian history in his book published in 1817?",
        "answer": "In 1817, James Mill published A History of British India and divided Indian history into three periods: Ancient, Medieval, and Modern."
    }
}

def load_preset(preset_name):
    if preset_name in PRESETS and preset_name != "Select a preloaded test case...":
        return PRESETS[preset_name]["question"], PRESETS[preset_name]["answer"]
    return "", ""

def update_api_key(key):
    if key and key.strip():
        evaluator.set_api_key(key.strip())
        return "✅ Gemini API Key configured successfully!"
    return "⚠️ Please enter a valid API Key."

def format_verdict_badge(verdict: str) -> str:
    verdict_clean = verdict.upper()
    if "FAITHFUL" in verdict_clean or "GROUNDED" in verdict_clean:
        return f'<div style="background-color:#0d9488; color:white; padding:12px; border-radius:8px; font-weight:bold; font-size:18px; text-align:center;">✅ VERDICT: {verdict}</div>'
    elif "PARTIAL" in verdict_clean:
        return f'<div style="background-color:#d97706; color:white; padding:12px; border-radius:8px; font-weight:bold; font-size:18px; text-align:center;">⚠️ VERDICT: {verdict}</div>'
    elif "HALLUCINAT" in verdict_clean:
        return f'<div style="background-color:#dc2626; color:white; padding:12px; border-radius:8px; font-weight:bold; font-size:18px; text-align:center;">❌ VERDICT: {verdict}</div>'
    else:
        return f'<div style="background-color:#4b5563; color:white; padding:12px; border-radius:8px; font-weight:bold; font-size:18px; text-align:center;">ℹ️ VERDICT: {verdict}</div>'

def process_chat_pipeline(question: str, custom_answer: Optional[str] = None):
    if not question or not question.strip():
        return "", "", "", "", "", ""

    # 1. Step: Generate or take candidate AI answer
    if custom_answer and custom_answer.strip():
        candidate_answer = custom_answer.strip()
    else:
        candidate_answer = evaluator.generate_candidate_answer(question)

    # 2. Step: Retrieve evidence from RAG Vector DB
    evidence_chunks = kb.retrieve_evidence(question, top_k=4)

    # 3. Step: Compare Evidence + Answer via LLM Evaluator
    eval_result = evaluator.evaluate(question, candidate_answer, evidence_chunks)

    # 4. Format outputs for UI
    verdict_html = format_verdict_badge(eval_result["verdict"])
    confidence = eval_result["confidence"]
    status = eval_result["status"]
    evidence = eval_result["evidence"]
    explanation = eval_result["explanation"]

    return candidate_answer, verdict_html, confidence, status, evidence, explanation

def upload_and_index_files(files):
    if not files:
        return "No files uploaded."
    messages = []
    for file_obj in files:
        count = kb.index_pdf(file_obj.name)
        fname = os.path.basename(file_obj.name)
        messages.append(f"Indexed '{fname}' ({count} chunks).")
    total_docs = kb.collection.count()
    return f"Successfully processed!\n" + "\n".join(messages) + f"\n\nTotal chunks in Knowledge Base: {total_docs}"


custom_css = """
.container { max-width: 1100px; margin: auto; }
.output-card { border-radius: 8px; padding: 12px; border: 1px solid #e5e7eb; background: #fafafa; }
"""

with gr.Blocks(title="Hallucination Detection in LLMs - NCERT History", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🛡️ Capstone Project: Hallucination Detection in LLMs
    ### RAG-Powered Evidence Verification & Factual Alignment for NCERT History & Civics
    
    This system implements the **Evidence-Answer Comparison Framework**:
    1. **User Input** (Question + AI Generated Answer)
    2. **RAG Vector Search** retrieves ground-truth passages from NCERT PDF chapters.
    3. **LLM-Based Evaluator** cross-compares claims against the retrieved evidence.
    4. Produces structured output: **Verdict**, **Confidence**, **Evidence**, **Status**, and **Explanation**.
    """)

    with gr.Accordion("⚙️ Configuration (Gemini API Key)", open=False):
        api_input = gr.Textbox(
            label="Google Gemini API Key",
            placeholder="Enter your Gemini API key (or set GEMINI_API_KEY env var)",
            type="password"
        )
        api_btn = gr.Button("Save API Key")
        api_status = gr.Markdown("")
        api_btn.click(update_api_key, inputs=[api_input], outputs=[api_status])

    with gr.Tabs():
        # TAB 1: Chatbot + Hallucination Guard
        with gr.Tab("💬 Automated Chatbot & Detection"):
            gr.Markdown("Ask any question from the NCERT History chapters. The AI generates an answer, which is immediately checked by the Hallucination Detection engine.")
            with gr.Row():
                with gr.Column(scale=5):
                    chat_question = gr.Textbox(
                        label="Enter Student Question",
                        placeholder="e.g., What did the Royal Charter of 1600 grant to the East India Company?",
                        lines=2
                    )
                    chat_submit_btn = gr.Button("🚀 Ask Chatbot & Detect Hallucinations", variant="primary")
                    chat_ai_answer = gr.Textbox(label="🤖 Generated AI Answer", lines=3, interactive=False)

                with gr.Column(scale=5):
                    chat_verdict_badge = gr.HTML(format_verdict_badge("AWAITING INPUT"))
                    with gr.Row():
                        chat_confidence = gr.Textbox(label="🎯 Confidence", interactive=False)
                        chat_status = gr.Textbox(label="📌 Status", interactive=False)
                    chat_evidence = gr.Textbox(label="📖 Retrieved NCERT Evidence", lines=4, interactive=False)
                    chat_explanation = gr.Textbox(label="💡 Evaluator Explanation", lines=4, interactive=False)

            chat_submit_btn.click(
                process_chat_pipeline,
                inputs=[chat_question, gr.State("")],
                outputs=[chat_ai_answer, chat_verdict_badge, chat_confidence, chat_status, chat_evidence, chat_explanation]
            )

        # TAB 2: Evaluator Test Bench (Question + AI Answer Comparison)
        with gr.Tab("🔬 Auditor Test Bench (Compare Question + Custom Answer)"):
            gr.Markdown("Directly evaluate candidate answers (e.g. from any LLM model) against NCERT ground truth. Use preloaded presets to test deliberate hallucinations.")
            
            preset_dropdown = gr.Dropdown(
                label="💡 Try a Preloaded Capstone Test Case (Hallucinations & Grounded Examples)",
                choices=list(PRESETS.keys()),
                value="Select a preloaded test case..."
            )
            
            with gr.Row():
                with gr.Column(scale=5):
                    bench_question = gr.Textbox(label="Input Question", lines=2)
                    bench_answer = gr.Textbox(label="Candidate AI Answer to Verify", lines=4)
                    bench_btn = gr.Button("🔍 Run Hallucination Evaluation", variant="primary")

                with gr.Column(scale=5):
                    bench_verdict_badge = gr.HTML(format_verdict_badge("AWAITING INPUT"))
                    with gr.Row():
                        bench_confidence = gr.Textbox(label="🎯 Confidence", interactive=False)
                        bench_status = gr.Textbox(label="📌 Status", interactive=False)
                    bench_evidence = gr.Textbox(label="📖 Retrieved NCERT Evidence", lines=4, interactive=False)
                    bench_explanation = gr.Textbox(label="💡 Evaluator Explanation", lines=5, interactive=False)

            preset_dropdown.change(
                load_preset,
                inputs=[preset_dropdown],
                outputs=[bench_question, bench_answer]
            )

            bench_btn.click(
                process_chat_pipeline,
                inputs=[bench_question, bench_answer],
                outputs=[bench_answer, bench_verdict_badge, bench_confidence, bench_status, bench_evidence, bench_explanation]
            )

        # TAB 3: Dataset & PDF Upload
        with gr.Tab("📚 NCERT Dataset & Indexing"):
            gr.Markdown("""
            ### Pre-indexed Dataset:
            - **Chapter 1**: *How, When and Where*
            - **Chapter 2**: *From Trade to Territory: The Company Establishes Power*
            - **Chapter 4**: *Tribals, Dikus and the Vision of a Golden Age*
            - **Chapter 6**: *Colonialism and the City: The Story of an Imperial Capital*
            
            You can also upload additional NCERT PDFs below to automatically expand the knowledge base.
            """)
            file_uploader = gr.File(label="Upload NCERT PDF(s)", file_types=[".pdf"], file_count="multiple")
            upload_btn = gr.Button("📥 Index Uploaded PDFs")
            upload_output = gr.Textbox(label="Indexing Status", lines=4)

            upload_btn.click(
                upload_and_index_files,
                inputs=[file_uploader],
                outputs=[upload_output]
            )

if __name__ == "__main__":
    demo.launch(share=True)
