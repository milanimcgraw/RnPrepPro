# Welcome to RNPrepPro! 🤖📚
### You're all-in-one NCLEX-RN study tool.📈

📥 In this repository, you'll find all of the information, files, and code to the RnPrepPro app. 

🕹️ RNPrepPro is an interactive Q&A tool that utilizes a comprehensive NCLEX study guide to generate Q&A pairs in order to aid furture nursing candidates pass the NCLEX-RN exam. The application will utilize a Retriever-Augmented Generation (RAG) model combined with fine-tuning of embeddings and a large language model (LLM) for interactive Q&A. 

**For more information on the the build process and the code, check out the **The Build Process** below!** 

## The Build Process ⚙️

* **Data:** [STUDYGUIDEZONE NCLEX Study Guide PDF](https://www.studyguidezone.com/images/nclexrnteststudyguide.pdf)
* **Models:** OpenAI text-3-embedding small, GPT-3.5-turbo, Mistral-7B-Instruct
* **Tooling:** LangChain with pymupdf parser & LLamaIndex with LlamaParser & reranker
* **Vector Store:** FAISS
* **Evaluation:** RAGAS (for all models), Eleuther AI LM Eval Harness (Mistral-7B-Instruct)

## CODE:
* [Langchain/LlamaIndex RAG w RAGAS Evaluation](https://colab.research.google.com/drive/1vOszspNFzd31HVLsExRyFEUQcwXcL4v_?usp=sharing)
* [Mistral-7B-Instruct-v2.0: Finetuning, RAG, & RAGAS Evaluation](https://colab.research.google.com/drive/1vOXjPyO09Aess3fxDCR5UJjJMeiNE4ox?usp=sharing) 


### PHASE 1 🛠️ 
* Generated a prompt to test with both Langchain and LLlamaIndex RAG to **answer 5 questions on 5 topics** from the data provided. 
* Generated a prompt to test with both Langchain and LLamaIndex RAG to **generated 5 Q&A pairs on 5 topics** from the data provided.
* Evaluate both RAG chains with RAGAS, and a prepared dataset.

### PHASE 2 🛠️
* **The Mistral-7B-Instruct-v0.2** was chosen as a smaller free model.
* The Mistral 7B model (based) was tested to **generate 5 Q&A pairs** with the same prompt and queries from PHASE 1.
* The model was then fine-tuned on a dataset from Hugging Face linked [here.](https://huggingface.co/datasets/openlifescienceai/medmcqa)
* The fine-tuned version was tested to **generate 5 Q&A pairs** with the same prompt and queries from PHASE 1.

### PHASE 3 🛠️
* Use the **Mistral model (base)** from PHASE 2 with RAG with the NCLEX Study Guide zone PDF. 
* Test it to generate Q&A pairs on the same prompt and queries from PHASE 1.
* Evaluate with RAGAS. 
* Use the **fine-tuned Mistral model** from PHASE 2 with RAG with the NCLEX Study Guide Zone PDF.
* Test it to generate Q&A pairs on the same prompt and queries from PHASE 1.
* Evaluate with RAGAS.

### PHASE 4 🛠️
* Compare each RAG chain and their outputs.
* Compare each RAGAS evaluation.
* Deploy the model with the RAG chain that performed the best on **Hugging Face using Chainlit and Docker**, based on evaluation results. 
