# Material, resources, overview of LLMs

- [General](#general)
- [LangChain](#langchain)
  - [DeepLearning.AI courses](#dlai-courses)
  - [Example Implementations / Guides](#example-implementations--guides)
- [Llama 2](#llama-2)
- [Embeddings](#embeddings)
- [Vector Database](#vector-database)
- [Deployment](#deployment)
- [Finetuning](#finetuning)
- [Other Material](#other-material)
  - [Prompt Injection](#prompt-injection)
  - [Educational](#educational)

# General

- [Open LLMs for Commercial Use](https://github.com/eugeneyan/open-llms)
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) (fast inference on most local models)
- [Huggingface Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Comparison of self-hosted LLMs and OpenAI: cost, text generation quality,
  development speed, privacy](https://betterprogramming.pub/you-dont-need-hosted-llms-do-you-1160b2520526)

## Langchain
- [ChatLangChain](https://chat.langchain.com/) (interactive documentation)
- [LangChain Hub](https://smith.langchain.com/hub) (hub for prompts)
- [LangSmith](https://smith.langchain.com/) (debugging, evaluating, monitoring 
  agents)

### DLAI courses
- [LangChain - LangChain for LLM Application Development](https://learn.deeplearning.ai/langchain/lesson/1/introduction)
- [LangChain - Chat with Your Data](https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/1/introduction)

The notebooks are in `DeepLearning.AI Notebooks`. **OpenAI API key is 
required** to run them.<br/><br/>
**Setup**:
- `conda env create -f environment.yml`
- `conda activate llm`
- `jupyter notebook`

*Objective*: run Llama 2 locally and make the notebooks work with it via 
API as well (where possible).<br/>

### Example Implementations / Guides

- [LangChain Local LLMs](https://python.langchain.com/docs/guides/local_llms)
- [LangChain + Streamlit + Llama Guide](https://ai.plainenglish.io/%EF%B8%8F-langchain-streamlit-llama-bringing-conversational-ai-to-your-local-machine-a1736252b172?gi=cfed6e717c75)
- [Llama v2.0, FAISS in Python using LangChain](https://webcache.googleusercontent.com/search?q=cache:https://medium.com/@mayuresh.gawai/implementation-of-llama-v2-in-python-using-langchain-%EF%B8%8F-ebebe82e881b)
- [Answering Question About your Documents Using LangChain](https://webcache.googleusercontent.com/search?q=cache:https://artificialcorner.com/answering-question-about-your-documents-using-langchain-and-not-openai-2f75b8d639ae)

**Without LangChain:**
-  [Building RAG from Scratch](https://gpt-index.readthedocs.io/en/latest/examples/low_level/oss_ingestion_retrieval.html) (with 
   LlamaIndex)

## Llama 2
- [How to prompt Llama 2](https://huggingface.co/blog/llama2#how-to-prompt-llama-2)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (also bindings for Python)
- [Quantize Llama models with GGML and llama.cpp](https://towardsdatascience.com/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172)
- [TinyLlama](https://github.com/jzhang38/TinyLlama) (still being trained)
- [Using LLaMA 2.0, FAISS and LangChain for Question-Answering](https://medium.com/@murtuza753/using-llama-2-0-faiss-and-langchain-for-question-answering-on-your-own-data-682241488476)

## Embeddings
- [Getting started with embeddings](https://huggingface.co/blog/getting-started-with-embeddings)
- [Chat with Anything - From PDFs Files to Image Documents](https://github.com/keitazoumana/Medium-Articles-Notebooks/blob/main/Chat_With_Any_Document.ipynb) (Notebook with examples on file type 
  detection, chunking, embedding)

## Vector Database
- [Chroma](https://www.trychroma.com/)
- [privateGPT](https://github.com/imartinez/privateGPT) (example of using Chroma)

## Deployment
- [LangChain Guide Deployment](https://python.langchain.com/docs/guides/deployments/)
- [LangChain Deployment Templates](https://python.langchain.com/docs/guides/deployments/template_repos)
- [Frameworks for Serving LLMs](https://betterprogramming.pub/frameworks-for-serving-llms-60b7f7b23407)

## Finetuning
- [DLAI Course Finetuning](https://learn.deeplearning.ai/finetuning-large-language-models/lesson/1/introduction)
- [Optimize open LLMs using GPTQ (run on consumer hardware)](https://www.philschmid.de/gptq-llama)
- [Digital Twins LLM](https://betterprogramming.pub/unleash-your-digital-twin-how-fine-tuning-llm-can-create-your-perfect-doppelganger-b5913e7dda2e?gi=2e25e4e85b76)
- [LangChain Finetune model in your voice (from chat data)](https://blog.langchain.dev/chat-loaders-finetune-a-chatmodel-in-your-voice/)

## Other Material

### Prompt Injection
- [What’s the worst that can happen?](https://simonwillison.net/2023/Apr/14/worst-that-can-happen/)
- [LLM Attacks](https://github.com/llm-attacks/llm-attacks)
 
### Educational
- [FCC Agile Software Development Handbook](https://www.freecodecamp.org/news/agile-software-development-handbook/)
- [Vicki Boykis: What Are Embeddings?](https://raw.githubusercontent.com/veekaybee/what_are_embeddings/main/embeddings.pdf)
- [Transformers Code step by step](https://towardsdatascience.com/nanogpt-learning-transformers-code-first-part-1-f2044cf5bca0)
- [Stanford NLU Spring 2023 Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp)
