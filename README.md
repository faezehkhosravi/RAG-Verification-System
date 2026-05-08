🤖 Multi-Agent RAG System for Policy Document Analysis
A modular, multi-agent Retrieval-Augmented Generation (RAG) pipeline built in Google Colab for intelligent analysis of OECD/IMF policy documents. The system ingests PDFs, processes and retrieves relevant content, generates structured summaries, runs policy debates, and verifies factual claims — all through a coordinated network of specialized AI agents.

📋 Table of Contents

Project Overview
System Architecture
Agents Description
Pipeline Flow
Technologies Used
Setup & Installation
How to Run
Output Files
Project Structure


🧠 Project Overview
This project implements a multi-agent NLP pipeline that:

Ingests and chunks PDF policy documents (OECD, IMF reports)
Preprocesses text using classical NLP (tokenization, NER, lemmatization)
Models topics using LDA and NMF
Generates Word2Vec and SBERT embeddings
Retrieves relevant documents using a Hybrid BM25 + Dense (Pinecone) retriever
Plans and generates RAG-based answers using Flan-T5
Summarizes documents into structured policy briefs
Runs a multi-agent policy debate (Pro-Innovation vs Precautionary)
Verifies factual claims with NLI and semantic similarity
Tracks memory, auto-tunes parameters, and visualizes system performance


🏗️ System Architecture

 
           PDF Documents
               │
               ▼
               

   ┌─────────────────────────┐
   
   │    PDFIngestionAgent    │
   
   │  Extracts & chunks PDFs │
   
   └────────────┬────────────┘
   
                │
                
                ▼
                
   ┌─────────────────────────┐
   
   │    PreprocessorAgent    │
   
   │ Tokenization, POS, NER, │
   
   │      Lemmatization      │
   
   └────────────┬────────────┘ 
               
              ┌──────┴──────┐
         
              │             │
         
              ▼             ▼
         
  ┌────────────────┐   ┌──────────────────────┐
 
  │   TopicModel   │         │    EmbeddingAgent    │
  
  │   (LDA / NMF)  │         │   Word2Vec / SBERT   │
 
  └────────────────┘   └──────────┬───────────┘
 
                                  │
  ```
                                  ▼

                      ┌──────────────────────┐
                      │    RetrieverAgent    │
                      │ Hybrid BM25 + Dense  │
                      │     (Pinecone)       │
                      └──────────┬───────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼

          ┌──────────────────┐    ┌──────────────────────────┐
          │   PlannerAgent   │    │ RetrieverExperimentAgent │
          │     (RAG + QA)   │    │ Reranking & Ablation     │
          └─────────┬────────┘    └──────────────────────────┘
                    │
                    ▼

          ┌──────────────────────┐
          │    SummarizerAgent   │
          │ Structured Policy    │
          │       Briefs         │
          └──────────┬───────────┘
                     │
             ┌───────┴────────┐
             ▼                ▼

     ┌────────────────┐   ┌──────────────────┐
     │  Debate Agents │   │  Verifier Agent │
     │ Pro vs Critical│   │ NLI Claim Check │
     └────────────────┘   └──────────────────┘
                     │
             ┌───────┴────────┐
             ▼                ▼

     ┌────────────────┐   ┌──────────────────┐
     │  Memory Agent  │   │ Visualizer Agent │
     │ Conversation & │   │ Confidence Plots │
     │ Retrieval Mem. │   │ + Agent Graph    │
     └────────────────┘   └──────────────────┘
```


🤖 Agents Description
1. pdf_ingestion_agent.py — PDF Ingestion Agent

Scans a Google Drive folder for all .pdf files
Extracts full text using pdfplumber
Splits documents into semantic chunks using LangChain's RecursiveCharacterTextSplitter
Outputs structured JSON with chunk metadata (source file, chunk ID, index)

2. preprocessor_agent.py — Preprocessor Agent

Takes ingested chunks and applies classical NLP using spaCy (en_core_web_sm)
Performs tokenization, POS tagging, lemmatization, and Named Entity Recognition (NER)
Outputs processed chunks with token/entity annotations

3. topic_model_agent.py — Topic Model Agent

Applies LDA (Latent Dirichlet Allocation) and NMF (Non-negative Matrix Factorization)
Uses CountVectorizer and TfidfVectorizer for feature extraction
Prints top keywords per topic for qualitative analysis

4. embedding_agent.py — Embedding Agent

Generates Word2Vec embeddings using Gensim
Generates SBERT embeddings using all-MiniLM-L6-v2 (SentenceTransformers)
Applies PCA + t-SNE for 2D visualization
Saves embeddings as .npy for downstream retrieval

5. retriever_agent.py — Retriever Agent

Implements Hybrid Retrieval: BM25 (sparse) + Pinecone (dense vector search)
Combines scores with a weighted alpha parameter (default: 0.6 dense / 0.4 sparse)
Saves retrieval diagnostics (top-k results, scores) to JSON

6. retriever_experiment_agent.py — Retriever Experiment Agent

Runs ablation experiments comparing Baseline Hybrid vs Cross-Encoder Reranking
Evaluates using Precision@K and Recall@K
Generates comparison plots saved to Google Drive

7. planner_agent.py — Planner Agent (RAG QA)

Loads retrieved documents and constructs a prompt with context
Runs RAG-based question answering using google/flan-t5-base
Saves the generated answer and source documents to JSON

8. summarizer_agent.py — Summarizer Agent

Generates structured summaries with sections: Overview, Key Policies, Governance Implications
Supports LLM-based and extractive fallback summarization
Outputs a StructuredSummary object with executive summary, key points, evidence with citations, and recommendations

9. debate_agent.py — Debate Agent

Simulates a policy debate between two AI agents:

Agent A (Pro-Innovation): argues for innovation-friendly regulation
Agent B (Precautionary): argues for safety-first regulation


Runs multiple rebuttal rounds and generates a joint conclusion
Saves the full transcript and consensus points to a text file

10. verifier_agent.py — Verifier Agent

Verifies factual claims against retrieved evidence
Uses NLI (Natural Language Inference) to check entailment/contradiction
Computes semantic similarity between claims and evidence
Checks temporal consistency of dated claims
Saves verification metrics to JSON

11. guardrails_agent.py — Guardrails Agent

Detects and redacts PII (emails, phone numbers, IBANs, SSNs)
Detects prompt injection attacks using regex patterns
Returns sanitized text and a list of security flags

12. memory_agent.py — Memory Agent

Logs experiment parameters (alpha, k, beta, latency, confidence) to a persistent JSON store
Implements auto-tuning via weighted k-NN grid search over past records
Suggests optimal hyperparameters based on historical performance

13. visualizer_agent.py — Visualizer Agent

Plots confidence trajectory over time from memory records
Builds and visualizes the agent interaction graph using NetworkX
Saves all plots to Google Drive


🔄 Pipeline Flow
Task 1: Ingestion + Preprocessing
  └─> classical_output.json

Task 2: Topic Modeling + Embeddings
  └─> sbert_embeddings.npy, embedding_map.png

Task 3: Hybrid Retrieval
  └─> retrieval_ablation.json

Task 4: RAG Question Answering
  └─> rag_qa_results.json

Task 5: Summarization + Debate
  └─> final_policy_brief.txt, summary.json

Task 6: Verification + Guardrails
  └─> metrics.json

Task 7: Memory + Visualization
  └─> memory.json, confidence_trajectory.png, agent_graph.png

🛠️ Technologies Used
CategoryLibrariesPDF Processingpdfplumber, LangChainClassical NLPspaCy (en_core_web_sm)Topic Modelingscikit-learn (LDA, NMF, TF-IDF)Embeddingsgensim (Word2Vec), sentence-transformers (SBERT)Vector DatabasePinecone (Serverless)Sparse Retrievalrank_bm25Rerankingcross-encoder/ms-marco-MiniLM-L-6-v2Language Modelsgoogle/flan-t5-base, google/flan-t5-smallVisualizationmatplotlib, networkxStorageGoogle Drive, JSONEnvironmentGoogle Colab

⚙️ Setup & Installation
1. Mount Google Drive in Colab
pythonfrom google.colab import drive
drive.mount('/content/drive')
2. Create the required folder structure in your Google Drive:
MyDrive/
├── data/           ← Put your PDF files here
├── results/        ← Output files will be saved here
│   └── plots/
└── queries/        ← Optional: policy_queries.json
3. Install dependencies
python!pip install pdfplumber langchain spacy sentence-transformers
!pip install rank_bm25 pinecone gensim transformers accelerate
!pip install networkx matplotlib pandas numpy scikit-learn
!python -m spacy download en_core_web_sm
4. Set API Keys
In retriever_agent.py, replace with your keys:
pythonPINECONE_API_KEY = "your-pinecone-api-key"
In planner_agent.py:
pythonHF_TOKEN = "your-huggingface-token"

▶️ How to Run
Run the agents in order (each task builds on the previous):
python# Task 1 — Ingestion & Preprocessing
run_task_1_pipeline()   # in preprocessor_agent.py

# Task 2 — Embeddings & Topic Modeling
# Run topic_model_agent.py then embedding_agent.py

# Task 3 — Retrieval
hybrid_retrieve(query)  # in retriever_agent.py

# Task 4 — RAG QA
generate_rag_answer(query, docs)  # in planner_agent.py

# Task 5 — Summarization & Debate
run_debate_and_save(agent_a, agent_b, ...)  # in debate_agent.py

# Task 6 — Verification
agent.verify_claims(claims, evidences)  # in verifier_agent.py

# Task 7 — Visualization
main()  # in visualizer_agent.py

📁 Output Files
FileDescriptionresults/classical_output.jsonPreprocessed chunks with NLP annotationsresults/sbert_embeddings.npySBERT embedding matrixresults/embedding_map.pngt-SNE visualization of embeddingsresults/retrieval_ablation.jsonHybrid retrieval resultsresults/retrieval_comparison.jsonReranking experiment resultsresults/rag_qa_results.jsonRAG question-answering outputresults/summary.jsonStructured policy summaryresults/final_policy_brief.txtFull debate transcript + consensusresults/metrics.jsonVerification & guardrails metricsresults/memory.jsonParameter tuning historyresults/plots/confidence_trajectory.pngConfidence over timeresults/plots/agent_graph.pngAgent interaction graph

📂 Project Structure
├── pdf_ingestion_agent.py       # Task 1: PDF loading & chunking
├── preprocessor_agent.py        # Task 1: NLP preprocessing
├── topic_model_agent.py         # Task 2: LDA & NMF topic modeling
├── embedding_agent.py           # Task 2: Word2Vec & SBERT embeddings
├── retriever_agent.py           # Task 3: Hybrid BM25 + Pinecone retrieval
├── retriever_experiment_agent.py# Task 3: Reranking ablation study
├── planner_agent.py             # Task 4: RAG-based QA with Flan-T5
├── summarizer_agent.py          # Task 5: Structured summarization
├── debate_agent.py              # Task 5: Policy debate simulation
├── verifier_agent.py            # Task 6: Claim verification (NLI)
├── guardrails_agent.py          # Task 6: PII redaction & prompt injection
├── memory_agent.py              # Task 7: Parameter logging & auto-tuning
└── visualizer_agent.py          # Task 7: Plots & agent graph

📌 Notes

All file paths point to Google Drive (/content/drive/MyDrive/...) — update these if running locally.
The system is modular: each agent can be used independently or as part of the full pipeline.
Replace placeholder API keys (Pinecone, HuggingFace) with your own before running.


👤 Author
Built as part of an NLP research project focused on automated policy document analysis using multi-agent AI systems.
