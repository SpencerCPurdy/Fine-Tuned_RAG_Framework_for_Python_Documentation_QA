# Fine-Tuned RAG Framework for Python Documentation Q&A

A Retrieval-Augmented Generation (RAG) system that answers questions about Python's standard library using a fine-tuned GPT-2 model with LoRA and vector search. This project demonstrates the complete pipeline of building a RAG system including data collection, model fine-tuning, vector database implementation, and evaluation.

## About

This portfolio project showcases practical skills in building RAG systems by implementing a question-answering system for Python documentation. The system combines semantic search over Python documentation with a fine-tuned language model to generate contextually relevant answers.

**Author:** Spencer Purdy  
**Development Environment:** Google Colab Pro (A100 GPU, High RAM)

## Features

- **Data Collection**: Automated scraping of Python 3 official documentation
- **Document Processing**: Chunking with overlap for optimal retrieval
- **Vector Database**: ChromaDB with sentence-transformers embeddings
- **Model Fine-Tuning**: GPT-2 fine-tuned using LoRA/PEFT for parameter efficiency
- **RAG Pipeline**: Combines retrieval and generation for grounded responses
- **Interactive Interface**: Gradio web application for querying the system
- **Comprehensive Evaluation**: ROUGE, BERTScore, and retrieval accuracy metrics
- **Performance Monitoring**: Tracks latency and sources retrieved per query

## Dataset

- **Source:** Python 3 Official Documentation (docs.python.org)
- **License:** Python Software Foundation License (PSF License, GPL-compatible)
- **Documents Collected:** 67
- **Total Chunks:** 5,257
- **Training Samples Generated:** 734

## System Performance

Performance metrics evaluated on test set:

| Metric | Score |
|--------|-------|
| Retrieval Accuracy | 94.0% |
| ROUGE-L F1 | 0.063 |
| BERTScore F1 | 0.794 |
| Average Latency | 2,084ms (~2 seconds) |
| Average Sources Retrieved | 1.2 per query |

**Model:** GPT-2 (124M parameters) with LoRA fine-tuning  
**Training Steps:** 500  
**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2

## Technical Stack

- **Model Fine-Tuning:** transformers, peft (LoRA)
- **Vector Database:** ChromaDB
- **Embeddings:** sentence-transformers
- **Evaluation Metrics:** rouge-score, bert-score
- **UI Framework:** Gradio
- **Data Processing:** beautifulsoup4, requests, pandas, numpy
- **Development:** Google Colab Pro with A100 GPU

## Setup and Usage

### Running in Google Colab

1. Clone this repository or download the notebook file
2. Upload `Fine-Tuned_RAG_Framework_for_Python_Documentation_Q_A.ipynb` to Google Colab
3. Select Runtime > Change runtime type > A100 GPU (or T4 GPU for free tier)
4. Run all cells sequentially

The notebook will automatically:
- Install required dependencies
- Collect Python documentation
- Process and chunk documents
- Fine-tune the GPT-2 model with LoRA
- Build the vector database
- Evaluate the system
- Launch a Gradio interface with a shareable link

### Running Locally

```bash
# Clone the repository
git clone https://github.com/SpencerCPurdy/finetuned-rag-python-docs.git
cd finetuned-rag-python-docs

# Install dependencies
pip install torch transformers datasets peft gradio pandas numpy scikit-learn tqdm requests beautifulsoup4 rouge-score bert-score accelerate sentence-transformers chromadb

# Run the notebook
jupyter notebook Fine-Tuned_RAG_Framework_for_Python_Documentation_Q_A.ipynb
```

**Note:** First run will take approximately 10-15 minutes for data collection, training, and setup.

## Project Structure

```
├── Fine-Tuned_RAG_Framework_for_Python_Documentation_Q_A.ipynb
├── README.md
├── LICENSE
└── .gitignore
```

The notebook contains the following components:

1. **Configuration & Setup**: System parameters, random seed initialization
2. **Data Collection**: Web scraping of Python documentation
3. **Document Processing**: Chunking and preprocessing
4. **Vector Database**: ChromaDB initialization and document embedding
5. **Model Fine-Tuning**: GPT-2 fine-tuning with LoRA
6. **RAG Pipeline**: Retrieval and generation integration
7. **Evaluation**: Comprehensive metrics computation
8. **Gradio Interface**: Interactive web application

## Key Implementation Details

- **Reproducibility:** All random seeds set to 42 for deterministic results
- **LoRA Configuration:** Rank 16, alpha 32, dropout 0.05
- **Chunk Size:** 400 characters with 50 character overlap
- **Retrieval:** Top-3 documents with minimum relevance score of 0.15
- **Generation Parameters:** Temperature 0.7, top-p 0.9, max 150 new tokens

## Limitations and Known Issues

### Data Limitations
- Limited to Python standard library documentation only (no third-party packages)
- Documentation snapshot may be outdated for latest Python versions
- Coverage of some modules may be incomplete

### Performance Limitations
- ROUGE-L F1 score of 0.063 indicates generated answers differ significantly from reference formats
- Answers can be verbose or include unnecessary details
- May generate plausible-sounding but incorrect information (hallucination)
- Sometimes fails to retrieve relevant sources for niche topics

### Input Limitations
- Maximum query length: 500 characters
- Best performance on clear, focused questions
- Ambiguous questions may produce generic answers

### General Limitations
- Not suitable for production use without further validation and testing
- Best for conceptual questions rather than version-specific details
- Always verify critical information with official Python documentation

## Evaluation Results

The system was evaluated on 50 test queries:

- **Retrieval Performance:** Successfully retrieved relevant documents for 94% of queries
- **Generation Quality:** BERTScore F1 of 0.794 indicates reasonable semantic similarity
- **Answer Format:** Low ROUGE scores suggest answers need better formatting and conciseness

## Example Queries

The system handles questions such as:
- "What is the datetime module used for?"
- "How do I read and write JSON files in Python?"
- "Explain list comprehensions in Python"
- "What are the main features of the collections module?"
- "How do I use regular expressions in Python?"
- "What is the difference between os and pathlib?"

## RAG Configuration

- **Chunk Size:** 400 characters
- **Chunk Overlap:** 50 characters
- **Retrieval Top-K:** 3 documents
- **Minimum Relevance Score:** 0.15
- **Vector Database:** ChromaDB with persistent storage

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Data Attribution

- **Python Documentation:** Python Software Foundation (PSF License)
- **GPT-2 Model:** OpenAI (MIT License)
- **Sentence-Transformers:** Apache 2.0 License

## Acknowledgments

- Python Software Foundation for excellent documentation
- Hugging Face for transformers and PEFT libraries
- Open-source community for the tools and frameworks used

## Contact

**Spencer Purdy**  
GitHub: [@SpencerCPurdy](https://github.com/SpencerCPurdy)

---

*This is a portfolio project developed to demonstrate RAG system implementation, model fine-tuning, and NLP engineering capabilities. The system is intended for educational and demonstrational purposes. Always verify important information with official Python documentation at https://docs.python.org/3/*
