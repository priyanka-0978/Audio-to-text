# Audio-to-text
**Internship Assignment for NLP(Voice AI)**

**Creation of for Speech to Text Model**

**Objective**

The primary goal of this internship assignment is to use natural language processing skills to develop a highly accurate transcription model for the Marathi language.

**Approach**

Downloaded the audio data and transcripted script(ground truth) provided for the task and created a folder 

Utilised hugging face openai/whisper-large-v3 Automatic speech recognition (ASR) and speech translation model. Because whisper models demonstrate a strong ability to generalise to many datasets and domains without the need for fine-tuning after being trained on 680k hours of labelled data.For speech recognition, the model predicts transcriptions in the same language as the audio. 

Whisper large-v3 is supported in Hugging Face Transformers through the main branch in the Transformers repo,so imported the necessary libraries: torch for PyTorch functionalities and AutoModelForSpeechSeq2Seq, AutoProcessor, and pipeline from the transformers library for loading the model and processing the input.

Configured the device accordingly. If a GPU is available, the CUDA device "cuda:0" is used; otherwise, the CPU is used.

Used AutoModelForSpeechSeq2Seq.from_pretrained() to load the model, passing the model_id, torch dtype, and use_safetensors=True to enable safe tensor usage.
Model.to(device) is used to move the model to the specified device.

A trained model, tokenizer, feature extractor, and other necessary parameters were used to start an automatic speech recognition pipeline. This pipeline would be used to tokenize and extract features from speech inputs before converting them to transcriptions or text output.

Created a list of transcriptions predicted by our model 

Converted the ground truth provided (.txt format) into list to calculate the accuracy of model using Word Error Rate metrics by comparing the hypothesis and ground truth 

Calculated the Word Error Rate for each predicted and actual text

Overall model performance was calculated by using average of WER 
 

