# DeepLearning.AI Notebooks

- based on:
  - [LangChain for LLM Application Development](https://learn.deeplearning.ai/langchain) 
  - [LangChain: Chat with Your Data](http://learn.deeplearning.ai/langchain-chat-with-your-data/)
</br></br>
- originally, an **OpenAI API key is required** to run the notebooks
- however, using [LocalAI](https://github.com/go-skynet/LocalAI) as an 
  **OpenAI drop-in alternative REST API**, you can use your own local LLM 
  (e.g. Llama2)

## Setup

### Install LocalAI

- follow [docs](https://localai.io/basics/getting_started/)
- e.g., for a [demo setup](https://localai.io/howtos/easy-setup-full/) on Windows:
```
# Make sure you have git, docker-desktop, and python 3.11 installed

git clone https://github.com/lunamidori5/localai-lunademo.git

cd localai-lunademo

call Setup.bat
```

### Setup environment

- you need `python3.10` otherwise requirements setup will fail
- make sure you are inside this directory `DeepLearning.AI Notebooks` and use either `conda` or `pip` setup

#### Conda
```
conda env create -f environment.yml

conda activate dlai
```

#### pip

```
# Windows

python -m venv dlai

.\dlai\Scripts\activate

pip install -r requirements.txt
```

## Usage

- after setting up LocalAI and environment, make sure your LocalAI docker 
  container is running with a model or use your OpenAI API key and call in 
  your CLI:
```
jupyter notebook
```
and navigate to the notebook you want to view.