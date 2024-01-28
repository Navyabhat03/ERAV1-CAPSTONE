# 🤖 Multi-modal LLM

Train a multi-modal LLM that takes text, image and audio as inputs.

<img width="332" alt="image" src="https://github.com/Navyabhat03/ERAV1-CAPSTONE/assets/60884505/6bc36b13-f231-4b2f-bc6e-69ed471b24fa">


🤗[**Space Link**](https://huggingface.co/spaces/Navyabhat/MultiModal-Phi2)

## Phi2 : Pretraining LLM from Scratch
### Details
1. Model used: **Microsoft Phi2** is a Python module that integrates language, image, and audio processing capabilities using advanced deep learning models. This module is particularly useful for creating applications that require understanding and generating responses based on multimodal inputs (text, image, and audio).
2. Dataset used: Tiny Stories dataset(100k samples) & Realtime data(100k samples) from finetuned Phi2 model via Ollama.
3. Pretraining approach: Pretraining using QLoRA.

### Training Logs
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/a6c143d0-c63c-4227-804f-93a4a8b74f7f)


## Phi2 : Multimodal Finetuning
### Details
1. LLM Backbone: Phi2
2. Vision Tower: clip-vit-large-patch14-336
3. Audio Model: Whisper
4. Pretraining Dataset: LAION-CC-SBU dataset with BLIP captions(200k samples)
5. Finetuning Dataset: Instruct 150k dataset based on COCO

### Overview
- **MultiModalPhi2** is a Python class for multimodal (text, image, and audio) input processing and generating text output. It utilizes models from transformers library and integrates image and audio processing capabilities along with text generation.

- **AudioLanguageConnector**: Connects audio processing outputs with language model inputs.

- **WhisperWithProjection**: Processes audio files and generates text transcriptions using Whisper model.

- **CLIPImageProcessor** is a class for processing images specifically for models trained with the CLIP architecture. CLIP (Contrastive Language–Image Pre-training) models are multimodal, meaning they understand both text and images. This class helps in pre-processing images so they can be input into a CLIP model.

### Requirements
To run this interface, you need to install certain Python packages. Ensure you have Python installed on your machine, and then install the required packages using pip:

[**Requirements**](https://github.com/Navyabhat03/ERAV1-CAPSTONE/blob/main/MultiModalPhi2/requirements.txt)

### Usage
#### Initialization
First, initialize the MultiModalPhi2 class:
```python
multimodal_phi2 = MultiModalPhi2()
```

This class is initialized with the following parameters:

- **modelname_or_path**: Path or name of the model.
- **temperature**: Temperature setting for response generation (default: 0.2).
- **max_new_tokens**: Maximum number of new tokens to generate (default: 1024).
- **device**: Device to use for computation (default: "cuda:0").

#### Processing Image
To process an image, use the PIL library to load the image and requests to fetch it from a URL:
```python
from PIL import Image
import requests

url = "https://upload.wikimedia.org/wikipedia/commons/0/0f/Grosser_Panda.JPG"
image = Image.open(requests.get(url, stream=True).raw)
```
#### Processing Audio
Load an audio sample using the datasets library:
```python
from datasets import load_dataset

audio_ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio = audio_ds[0]["audio"]
```

#### Generating Output
To generate output, pass text, audio, and image data to the multimodal_phi2 instance:
```python
text = "Tell me about this scene"
output = multimodal_phi2(text, audio, image)
print(output)

```

#### Methods
- **disable_torch_init**: Disables redundant torch default initializations to speed up model creation.
- **load_pretrained_model**: Loads the pretrained model, tokenizer, and image processor.
- **tokenizer_image_token**: Processes and tokenizes the input prompt, particularly handling image tokens.
- **__call__**: The main method to process the input (text, audio, image) and generate an output.

### Functionality
- **Image Processing**: The model processes images using the CLIPImageProcessor from the transformers library.
- **Audio Processing**: Audio data is processed using WhisperProcessor and WhisperForConditionalGeneration from transformers.
- **Text Generation**: The class uses the LlavaPhiForCausalLM model for generating text based on multimodal inputs (text, image, and audio).

### Customization
You can customize various parameters such as model name, temperature, and max tokens during the initialization of *MultiModalPhi2*.

### Limitations
- The effectiveness depends on the quality and relevance of input data.
- The model size and computational requirements might be significant due to the complexity of multimodal processing.


### Pretraining
#### Training Logs
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/76543d98-d9fe-4c1a-ac47-3d06e48053ad)

### Finetuning
#### Training Logs
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/2747acce-bc99-4c37-a05a-d5e81cb9aa9d)

### Results
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/f12a9f04-df32-413e-b957-774c30381b2b)

### Deployed on HF
#### Text & Image:
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/485a2806-81ac-4229-97ee-87f58af578bc)
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/ae2c14c4-6949-4fff-b2fb-cb37a29eac33)

#### Audio & Image:
**Question Asked: How many people are there in this image?**
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/430310fc-1df9-459c-94f3-32d9691a1035)
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/fd30a864-b289-469a-9c85-b6fd02f486a9)
On HF Space:
![image](https://github.com/RaviNaik/ERA-CAPSTONE/assets/23289802/efefee6e-98ee-4658-b2e9-f18d8f82a234)

