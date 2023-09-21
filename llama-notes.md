
# Base Models
## Hardware Requirements 

> To load a model in full precision, i.e. 32-bit (or float-32) on a GPU for 
> downstream training or inference, it costs about 4GB in memory per 1 billion 
> parameters. So, just to load Llama-2 at 70 billion parameters, it costs 
> around 280GB in memory at full precision.
> 
> Edit: Llama-2 is actually published in 16bit not 32bit (although many LLMs 
> are published in 32bit). The math remains the same regardless, it would 
> cost 180GB to load Llama-2 70B.
> 
> Now, there is the option to load models at different precision levels (at 
> the sacrifice of performance). If you load in 8-bit, you will incur 1GB of 
> memory per billion parameters, which would still require 70GB of GPU memory 
> for loading in 8-bit.

[Source](https://webcache.googleusercontent.com/search?q=cache:https://pub.aimind.so/this-is-why-you-cant-use-llama-2-d33701ce0766)</br></br>

### Llama 1
> To effectively use the models, it is essential to consider the memory and 
> disk requirements. Since the models are currently loaded entirely into 
> memory, you will need sufficient disk space to store them and enough RAM to 
> load them during execution. When it comes to the 65B model, even after 
> quantization, it is recommended to have at least 40 gigabytes of RAM 
> available. It’s worth noting that the memory and disk requirements are 
> currently equivalent.

[Source](https://ai.plainenglish.io/%EF%B8%8F-langchain-streamlit-llama-bringing-conversational-ai-to-your-local-machine-a1736252b172)</br></br>


### Llama-2-7b-chat-hf

> You will need ~8GB of GPU RAM for inference and running on CPU is practically
> impossible.

[Source](https://medium.com/@murtuza753/using-llama-2-0-faiss-and-langchain-for-question-answering-on-your-own-data-682241488476)</br></br>

> The model you use will vary depending on your hardware. For good results, you 
> should have at least 10GB VRAM at a minimum for the 7B model, though you can 
> sometimes see success with 8GB VRAM. The 13B model can run on GPUs like the 
> RTX 3090 and RTX 4090. The largest model, however, will require very powerful 
> hardware like an A100 80GB.

[Source](https://easywithai.com/resources/llama-2/)</br></br>

# Quantized Models

> In a nutshell:<br>
> Quantized models are basically compressed or "shrunken" versions, easier to 
> run if you don't have strong hardware (and is also easier on storage).<br><br>
> They usually perform slightly worse than their unquantized versions: the 
> lower the quant, the worse it gets (although 8 bit is almost if not just as 
> good as its unquantized version). 4 bit seems to be the best compromise 
> between performance and size/speed, but we now have 5 bit, 6 bit, 2 bit, etc.
> you can choose from depending on your need.<br><br>
> Another rule is while the lower a quantization is, parameter count is still 
> better. So a model with lots of parameters but a low quant will still perform
> better than one with a higher quant (or even unquantized) but less 
> parameters. So in theory, a quantized 13b will work better than an 
> unquantized 7b if I'm right.

[Source](https://www.reddit.com/r/LocalLLaMA/comments/15zz81s/llama2_quantized_model_vs_regular_one_whats_the/)

## Hardware Requirements

> The best technique depends on your GPU: if you have enough VRAM to fit the 
> entire quantized model, GPTQ with ExLlama will be the fastest. If that’s 
> not the case, you can offload some layers and use GGML models with llama.cpp
> to run your LLM.

[Source](https://towardsdatascience.com/quantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172)</br></br>