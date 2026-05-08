
import spacy
from typing import List, Dict, Any

# Load the spaCy model once
NLP_MODEL = spacy.load("en_core_web_sm")

class PreprocessorAgent:
    """
    Agent responsible for classical NLP pre-processing tasks
    like tokenization, POS tagging, lemmatization, and NER.
    """
    def __init__(self, nlp_model: Any = NLP_MODEL):
        """Initializes the agent with the pre-loaded spaCy model."""
        self.nlp = nlp_model
        if not self.nlp:
            raise ValueError("SpaCy model not loaded successfully.")

    def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Performs classical NLP tasks on a single text chunk."""

        doc = self.nlp(chunk["original_text"])

        # 1. Tokenization, POS, and Lemmatization
        tokens_data = [
            {
                "token": token.text,
                "pos": token.pos_,
                "lemma": token.lemma_
            }
            for token in doc
        ]

        # 2. Named Entity Recognition (NER)
        entities_data = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char
            }
            for ent in doc.ents
        ]

        # Combine results with original metadata
        processed_chunk = {
            "chunk_id": chunk["metadata"]["chunk_id"],
            "source_file": chunk["metadata"]["source_file"],
            "original_text": chunk["original_text"],
            # Placeholder for redaction, to be refined in Task 6
            "redacted_text": chunk["original_text"],
            "tokens": tokens_data,
            "entities": entities_data,
        }

        return processed_chunk

    def preprocess_data(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a list of chunks from the Ingestion Agent."""
        print(f"\n[PreprocessorAgent] Starting classical NLP processing on {len(chunks)} chunks...")
        processed_data = [self.process_chunk(chunk) for chunk in chunks]
        print("[PreprocessorAgent] Classical NLP processing complete.")
        return processed_data

import os
import json
import pdfplumber
from typing import List, Dict, Any


def run_task_1_pipeline():
    """Executes the sequential workflow for Task 1 (Ingestion -> Preprocessing)."""
    print("\n=======================================================")
    print("--- Starting Task 1: Ingestion & Pre-Processing Pipeline ---")
    print("=======================================================")

    # 1. Initialize Agents
    ingestion_agent = PDFIngestionAgent(chunk_size=1000, chunk_overlap=200)

    try:
        preprocessor_agent = PreprocessorAgent()
    except ValueError as e:
        print(f"FATAL ERROR: {e}")
        return

    # 2. Step 1: Ingestion and Chunking
    ingested_chunks = ingestion_agent.ingest_pdfs(data_dir=DATA_DIR)

    if not ingested_chunks:
        print("Pipeline aborted: No chunks to process.")
        return

    # 3. Step 2: Classical NLP Pre-Processing
    final_output = preprocessor_agent.preprocess_data(chunks=ingested_chunks)

    # 4. Save Final Output
    try:
        OUTPUT_PATH = "/content/drive/MyDrive/results/classical_output.json"
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        print(f"\n=======================================================")
        print(f"✅ Task 1 Complete. Total {len(final_output)} final records saved.")
        print(f"Output saved to: {OUTPUT_PATH}")
        print("=======================================================")
    except Exception as e:
        print(f"Error saving file: {e}")

run_task_1_pipeline()