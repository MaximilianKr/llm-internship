
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

