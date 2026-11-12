# 🛡️ Capstone Project: Hallucination Detection in Large Language Models (LLMs)

### Domain: NCERT History & Civics Knowledge Grounding
### Architecture: RAG-Retrieved Evidence + Candidate Answer Comparison via LLM Evaluator

---

## 📌 1. Project Overview & Methodology

Large Language Models (LLMs) frequently suffer from **hallucinations**—generating responses that sound fluent and authoritative but are factually incorrect or ungrounded. In academic domains such as school textbooks (e.g., NCERT History and Civics), hallucinations mislead learners.

This project implements the **Evidence-Answer Comparison Framework** designed specifically to detect, classify, and explain hallucinations in real time.

### 🔄 System Architecture (Flowchart)

```text
                       ┌──────────────────────────────┐
                       │          User Input          │
                       │     (Question + AI Answer)   │
                       └──────────────┬───────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
             ┌───────────┐                    ┌─────────────────────┐
             │ Question  │                    │ AI-generated Answer │
             └─────┬─────┘                    └──────────┬──────────┘
                   ▼                                     │
       ┌────────────────────────┐                        │
       │   Retrieve Evidence    │                        │
       │   (RAG / Vector DB)    │                        │
       └───────────┬────────────┘                        │
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │ Evidence + Answer Comparison  │
                      └───────────────┬───────────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │      LLM-Based Evaluator      │
                      └───────┬───┬───┬───┬───┬───────┘
                              │   │   │   │   │
         ┌────────────────────┘   │   │   │   └────────────────────┐
         ▼                        ▼   ▼   ▼                        ▼
    ┌─────────┐    ┌────────────┐ ┌────────┐ ┌────────┐    ┌─────────────┐
    │ Verdict │    │ Confidence │ │Evidence│ │ Status │    │ Explanation │
    └─────────┘    └────────────┘ └────────┘ └────────┘    └─────────────┘
```

---

## 📊 2. Evaluator Output Schema

For every candidate answer, the LLM-based evaluator produces 5 structured outputs:

| Output Field | Description | Example Values |
|---|---|---|
| **Verdict** | Final categorical judgment on the factual accuracy of the answer | `FAITHFUL / GROUNDED`, `HALLUCINATED`, `PARTIALLY HALLUCINATED` |
| **Confidence** | Statistical / Evaluator certainty score | `95%`, `90%`, `80%` |
| **Evidence** | Direct verbatim quote(s) and chapter/page citation from the NCERT text | *"In 1757, Robert Clive led the Company's army against Sirajuddaulah at Plassey." (Ch 2, p. 12)* |
| **Status** | Specific factual relationship between the answer and evidence | `Verified / Supported`, `Contradicted`, `Unsupported / Fabricated` |
| **Explanation** | Step-by-step reasoning explaining which parts of the claim are false or verified | *"The answer incorrectly asserts that Warren Hastings led the battle in 1857. In reality, Robert Clive led the Battle of Plassey in 1757..."* |

---

## 📂 3. Pre-Indexed Dataset (NCERT History: *Our Pasts - III*)

The system comes pre-indexed with Class 8 NCERT History chapters:
- **Chapter 1**: *How, When and Where* (James Rennel, James Mill's 1817 periodisation, British official records).
- **Chapter 2**: *From Trade to Territory: The Company Establishes Power* (1600 Royal Charter, Battle of Plassey 1757, Battle of Buxar 1764, Diwani of Bengal 1765, Subsidiary Alliance, Tipu Sultan, Doctrine of Lapse).
- **Chapter 4**: *Tribals, Dikus and the Vision of a Golden Age* (Birsa Munda movement, Mundas, Santhals, Jhum shifting cultivation, colonial forest laws).
- **Chapter 6**: *Colonialism and the City: The Story of an Imperial Capital* (De-urbanisation of Machlipatnam and Surat, Shahjahanabad, Delhi Renaissance, Lutyens and Baker New Delhi design).

---

## 🚀 4. How to Run on Google Colab (Recommended)

1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Click **File -> Upload notebook** and select `hallucination_detection_colab.ipynb`.
3. In Colab, click **Runtime -> Run all** (or run cells one by one).
4. Enter your free **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey).
5. The final cell will launch the **Gradio Web Interface** directly inside Colab and provide a public `.gradio.live` link that you can share with evaluators or project supervisors!

---

## 💻 5. How to Run Locally

```bash
# 1. Clone or navigate to the directory
cd hallucination_detection_rag

# 2. Install requirements
pip install -r requirements.txt

# 3. Set your Gemini API key (Windows PowerShell)
$env:GEMINI_API_KEY="your-gemini-api-key"

# 4. Launch the application
python app.py
```
Open `http://localhost:7860` in your web browser.

---

## 🧪 6. Demonstration & Capstone Evaluation Test Cases

### Test Case 1: Direct Hallucination (Historical Distortion)
- **Question**: *"When was the Battle of Plassey fought and who led the British forces?"*
- **Candidate AI Answer**: *"The Battle of Plassey was fought in 1857. The British army was led by Warren Hastings who defeated Tipu Sultan with the help of the French navy."*
- **Evaluation Output**:
  - **Verdict**: ❌ `HALLUCINATED`
  - **Confidence**: `98%`
  - **Status**: `Contradicted`
  - **Evidence**: *"In 1757, Robert Clive led the Company's army against Sirajuddaulah at Plassey." (NCERT Ch 2, p. 12)*
  - **Explanation**: The answer contains multiple severe factual hallucinations: the year was 1757, not 1857; the commander was Robert Clive, not Warren Hastings; the opponent was Sirajuddaulah of Bengal, not Tipu Sultan; and the French were rivals, not allies.

---

### Test Case 2: Grounded & Faithful Answer
- **Question**: *"Who devised the Doctrine of Lapse and which kingdoms were annexed under it?"*
- **Candidate AI Answer**: *"Lord Dalhousie devised the Doctrine of Lapse. Under this doctrine, kingdoms like Satara, Sambalpur, Udaipur, Nagpur, and Jhansi were annexed."*
- **Evaluation Output**:
  - **Verdict**: ✅ `FAITHFUL / GROUNDED`
  - **Confidence**: `99%`
  - **Status**: `Verified / Supported`
  - **Evidence**: *"The final wave of annexations occurred under Lord Dalhousie... One kingdom after another was annexed: Satara (1848), Sambalpur (1850), Udaipur (1852), Nagpur (1853) and Jhansi (1854)." (NCERT Ch 2, p. 19)*
  - **Explanation**: All claims accurately match the textbook facts without omissions or fabrications.

---

### Test Case 3: Partial Hallucination (Subtle Attribution Error)
- **Question**: *"How did James Mill divide Indian history in his book published in 1817?"*
- **Candidate AI Answer**: *"In 1817, James Mill published A History of British India and divided Indian history into Ancient, Medieval, and Modern periods."*
- **Evaluation Output**:
  - **Verdict**: ⚠️ `PARTIALLY HALLUCINATED`
  - **Confidence**: `94%`
  - **Status**: `Contradicted`
  - **Evidence**: *"In 1817, James Mill published A History of British India. In this he divided Indian history into three periods - Hindu, Muslim and British. Later historians divided Indian history into 'ancient', 'medieval' and 'modern'." (NCERT Ch 1, p. 3-4)*
  - **Explanation**: The book title and year are accurate, but Mill periodised history into "Hindu, Muslim, and British". The division into "Ancient, Medieval, and Modern" was done by later historians.

---

## 🎓 7. Capstone Defense Viva Questions & Answers

1. **Q: What is the difference between Intrinsic and Extrinsic Hallucinations?**
   - *Intrinsic Hallucination*: The generated response directly contradicts the retrieved source context (e.g., claiming Dalhousie was in 1757 when the source says 1848).
   - *Extrinsic Hallucination*: The response adds details that cannot be verified or found in the source text (e.g., claiming a French submarine was present).

2. **Q: Why use RAG instead of just prompting the LLM directly?**
   - General LLMs are trained on general internet data and can conflate similar events across history. RAG restricts the evaluation domain strictly to verified, ground-truth NCERT textbooks, establishing a closed-world knowledge boundary.

3. **Q: How does the Evaluator prevent evaluating its own biases?**
   - The evaluator is decoupled from answer generation. It uses strict few-shot verification instructions requiring exact verbatim quotations from the retrieved chunks before making a verdict.
