# 🤖 Multi-modal LLM

This is a multi-modal LLM that takes text, image and audio as inputs.

<img width="332" alt="image" src="https://github.com/Navyabhat03/ERAV1-CAPSTONE/assets/60884505/6bc36b13-f231-4b2f-bc6e-69ed471b24fa">


🤗[**Space Link**](https://huggingface.co/spaces/Navyabhat/Capstone_Project)

## Phi2 : Pretraining LLM from Scratch
### Details
1. Model used: **Microsoft Phi2** is a Python module that integrates language, image, and audio processing capabilities using advanced deep learning models. This module is particularly useful for creating applications that require understanding and generating responses based on multimodal inputs (text, image, and audio).
2. Dataset used: Tiny Stories dataset(100k samples) & Realtime data(100k samples) from finetuned Phi2 model via Ollama.
3. Pretraining approach: Pretraining using QLoRA.

### Training Logs
![image](https://github.com/Navyabhat03/ERAV1-CAPSTONE/assets/60884505/050c6151-4365-4e37-8591-a3a1c6674ead)

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

### Pretraining
#### Training Logs
![image](https://github.com/Navyabhat03/ERAV1-CAPSTONE/assets/60884505/11be0ba9-6bd0-47a1-90a6-4a4b4c3314fa)

### Finetuning
#### Training Logs
![image](https://github.com/Navyabhat03/ERAV1-CAPSTONE/assets/60884505/800fef1d-a97b-4014-b505-42a53284e748)

### Results
#### Text & Image:
![Screenshot (374)](https://github.com/Navyabhat03/ERAV1-CAPSTONE/assets/60884505/019dad79-c6db-424e-a7a7-bed6a09612aa)

![Screenshot (373)](https://github.com/Navyabhat03/ERAV1-CAPSTONE/assets/60884505/12dbc4c5-bd0f-4694-8a85-934f46d9cab9)

#### Audio & Image:
**Question Asked: Describe the image in one line**
![Screenshot (375)](https://github.com/Navyabhat03/ERAV1-CAPSTONE/assets/60884505/a252eca5-d764-4614-96e6-3595538e2cdd)

### Customization
You can customize various parameters such as model name, temperature, and max tokens during the initialization of *MultiModalPhi2*.

### Limitations
- The effectiveness depends on the quality and relevance of input data.
- The model size and computational requirements might be significant due to the complexity of multimodal processing.
