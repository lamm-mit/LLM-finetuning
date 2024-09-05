# LLM Finetuning for domain adaptation

Various strategies are presented to fine-tune LLMs, using CPT, SFT and DPO/ORPO. This repositories includes all codes, some based on the Hugging Face Alignment Handbook (https://github.com/huggingface/alignment-handbook). A snapshot of this code is included in the subdirectory ```alignment-handbook``` (as of Sept. 1, 2024). Note, several of the training Python files (e.g. ```run_cpt.py```, ```run_sft.py```, etc. have been updated; correct versions are included in the root directory). 

![image](https://github.com/user-attachments/assets/a9514ac2-849b-4678-8580-0f6e40f219fb)

## Training 

### Step 1: Install Hugging Face Alignment Handbook code

```
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
python -m pip install .
```

### Step 2: Train models

Each of the folders include YAML configs for the individual models. From each directory, the whole training series is executed like so:

```
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_cpt.py config_full_CPT-Base.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_sft.py config_full_SFT-Base.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_orpo.py config_full_ORPO-Base.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_dpo.py config_full_DPO-Base.yaml

ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_cpt.py config_full_CPT-Instruct.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_sft.py config_full_SFT-Instruct.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_orpo.py config_full_ORPO-Instruct.yaml
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file deepspeed_zero3.yaml run_dpo.py config_full_DPO-Instruct.yaml
```
Check to make sure the input and output files in each of the YAML files are correct. E.g., the CPT run should produce the input to the SFT run, and the SFT run should produce the input to ORPO/DPO. 


## Model Merging 

```
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .  # install the package and make scripts available
```

```python
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random

from tqdm.notebook import tqdm
!wandb login <WANB_TOKEN>
token = 'hf_XXXXXXXXXXXXXXXXXXXXX'
from huggingface_hub import login
login(token=token)
import torch
torch.cuda.get_device_name(), torch.cuda.device_count()
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)
    
from accelerate import infer_auto_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from datasets import load_dataset
from datasets import IterableDataset

from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import TrainerCallback
from transformers import AutoConfig
from transformers import BitsAndBytesConfig

from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import multiprocessing
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

device = torch.device("cuda")
torch.version.cuda, torch.__version__, torch.backends.cudnn.version(), torch.backends.cudnn.enabled, torch.cuda.is_available()

def load_model (model_name='lamm-mit/Llama3.1-8b-Instruct-CPT-SFT-DPO-09022024', chat_template=None,
                #compile_mode="max-autotune", 
                compile_mode=None,
                attn_implementation="flash_attention_2",
                quant=False):
     
    bnb_config4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        #bnb_4bit_use_double_quant=True,
    )

    bnb_config4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if quant==False:
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # revision='1a02ed84a2b3525452f7cffd0c1002cc657bd3e9', #epoch 1
            trust_remote_code=True,
            use_cache=False,
            device_map="auto",
            torch_dtype =torch.bfloat16,
            attn_implementation=attn_implementation,
        ) 
        #model.config.use_cache = False
    else:
        print ("Use quantized model.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # revision='xxxxxxxxxxxxxxxxxxxxx', 
            trust_remote_code=True,
            use_cache=False,
            quantization_config =bnb_config4bit,
            device_map="auto",
            torch_dtype =torch.bfloat16,
            attn_implementation=attn_implementation,
        )

    if compile_mode != None: 
        model.generation_config.cache_implementation = "static"
        model.forward = torch.compile(model.forward, mode=compile_mode, fullgraph=True )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_bos_token=False
                                             )
    if chat_template != None:
        print("Set custom chat template.")
        tokenizer.chat_template=chat_template

    eos_token=tokenizer.eos_token_id
    eos_token 
    return model, tokenizer

try:
    del model
    del tokenizer
except:
    print ()
    
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_SLERP_and_save (source='lamm-mit/mistral-7B-v0.3-Base-CPT', 
                    base_model='lamm-mit/mistralai-Mistral-7B-Instruct-v0.3',
                    upload_to_hub=False, private=True,
                    attn_filter = [0, 0.5, 0.3, 0.7, 1],
                    mlp_filter  = [1, 0.5, 0.7, 0.3, 0],
                    hub_id='lamm-mit',modifier='',
                       ):


    source_base = source.split('/')[-1]
    MODEL_NAME = source_base + f"_SLERP{modifier}"
     
    yaml_config = f"""
    slices:
      - sources:
          - model: {source}
            layer_range: [0, 32]
          - model: {base_model}
            layer_range: [0, 32]
    merge_method: slerp
    base_model: {base_model}
    parameters:
      t:
        - filter: self_attn
          value: {attn_filter}
        - filter: mlp
          value: {mlp_filter}
        - value: 0.5
    dtype: bfloat16
    
    """
    
    # Save config as yaml file
    with open('config.yaml', 'w', encoding="utf-8") as f:
        f.write(yaml_config)
         
    !mergekit-yaml --cuda --copy-tokenizer --allow-crimes --out-shard-size 5B --lazy-unpickle config.yaml {MODEL_NAME}

    if upload_to_hub:
        print ("Pushing to hub...")
        model, tokenizer,=load_model (model_name=MODEL_NAME)
        
        tokenizer.push_to_hub (hub_id+'/'+MODEL_NAME, private=private)
        model.push_to_hub (hub_id+'/'+MODEL_NAME, private=private)
    
        try:
            del model
            del tokenizer
        except:
            print ()

        print ("HF hub model name: ", hub_id+'/'+MODEL_NAME)
            
    return MODEL_NAME
```

Then, you can merge in this way:

```python
model_name=merge_SLERP_and_save (source='lamm-mit/mistral-7B-v0.3-Instruct-CPT', 
                    base_model='lamm-mit/mistralai-Mistral-7B-Instruct-v0.3',
                    upload_to_hub=True, #modifier='_var_H',
                    attn_filter =[0, 0.5, 0.3, 0.7, 1],
                    mlp_filter= [1, 0.5, 0.7, 0.3, 0],
                    #attn_filter =[0, 0.3, 0.5, 0.7, 1],
                    #mlp_filter= [1, 0.7, 0.5, 0.3, 0],
                       )
#If you need to manually copy/push tokenizer, e.g. for Mistral models 
tokenizer = AutoTokenizer.from_pretrained('lamm-mit/mistralai-Mistral-7B-Instruct-v0.3', trust_remote_code=True, add_bos_token=False)
tokenizer.push_to_hub ('lamm-mit/'+model_name, private=True)
```
![image](https://github.com/user-attachments/assets/0458583a-df8e-4460-9b52-8ed9dce383df)

## Inference 

```python
def generate_response (model,tokenizer,text_input="Biology offers amazing",system_prompt='You are a materials scientist.',
                      num_return_sequences=1,
                      temperature=0.3,  
                      max_new_tokens=256,do_sample=True,
                      num_beams=1,eos_token_id= [
                                            #128001, 128008, 128009 # for Llama-3.1 models
                                            2, #Mistral models 
                                            #1,107, #for Gemma models
                                          ],device='cuda',
                      top_k = 50, top_p =0.9, repetition_penalty=1.,
                      messages=[], verbatim=False,
                      ):

    if messages==[]:
        if system_prompt!='':
            messages=[    {"role": "system", "content": system_prompt},
                          {"role": "user", "content": text_input}]
        else:
                messages=[{"role": "user", "content": text_input}]
    else:
        messages.append (#{"role": "system", "content":system_prompt},
                          {"role": "user", "content":text_input})
        
    text_input = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    if verbatim:
        print (text_input)
    inputs = tokenizer([text_input],  add_special_tokens = False,  return_tensors ='pt' ).to(device)
    with torch.no_grad():
          outputs = model.generate(**inputs, #input_ids=inputs.to(device), 
                                   max_new_tokens=max_new_tokens,
                                   temperature=temperature, #value used to modulate the next token probabilities.
                                   num_beams=num_beams, top_k = top_k,do_sample=do_sample,
                                   top_p =top_p,eos_token_id=eos_token_id,
                                   num_return_sequences = num_return_sequences, repetition_penalty=repetition_penalty,
                                  )

    outputs=outputs[:, inputs["input_ids"].shape[1]:]

    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True), messages   
```
Loading and sampling model:
```python
try:
    del model
    del tokenizer
except:
    pass
    
model, tokenizer,=load_model (model_name='lamm-mit/Llama3.1-8b-Instruct-CPT-SFT-DPO-09022024')

messages=[]
result,messages=generate_response (model, tokenizer, text_input="Collagen and leaves, discuss their relationship.",num_return_sequences=1, 
                                   system_prompt="You are a materials scientist.",
                                   temperature=0.1,max_new_tokens=256 , messages=[], verbatim=False )

for res in result:
    print(res, "\n")

messages.append (  {"role": "assistant", "content":result[0]})  
```

### Systematic prompting with pre-determined multi-turn interactions

This code runs a pre-programmed multi-turn interaction with a set of models. The code produces .MD markup files that can optionally be converted to a LaTeX box. 

```python
import time
import os
import os
import glob
import re

def sanitize_filename(filename):
    # Replace slashes and other special characters with underscores
    return re.sub(r'[\/:*?"<>|]', '_', filename)

# Dictionary associating model names
model_associations = {
    "lamm-mit/Llama3.1-8b-Instruct-CPT-SFT-DPO-09022024": "lamm-mit/Llama3.1-8b-Instruct-CPT-SFT_DPO",
    "lamm-mit/Llama3.1-8b-Instruct-CPT-SFT-ORPO-SLERP-09022024": "lamm-mit/Llama3.1-8b-Instruct-CPT-SFT-ORPO-SLERP",
    "lamm-mit/SmolLM-Base-1.7B-CPT-SFT-DPO-09022024": "lamm-mit/SmolLM-Base-1.7B-CPT-SFT-DPO",
}

eos_token_id_Llama= [ 128001, 128008,128009, ]
eos_token_id_Mistral= [ 2, ]
                                     
system_prompt=("Your are a helpful and friendly assistant, and a creative expert in materials science with special knowledge in biological materials. You provide concise but accurate responses. "
               "You are a highly capable image generation prompt maker. You excel in combining disparate concepts and "
               "develop well thought-out prompts for algorithms like DALLE-3 or Stable Diffusion. "
               "Keep the prompts short but detailed enough to render complex images of the materials. "
               "Instead of using technical terms, you carefully describe the image using generic features such as "
               "lines, shapes, circles, holes, patterns, and so on. "
               "inspired by microscopic images of leaves. ")

questions=["We will develop an image description step by step. First, think about design principles you can extract from combining spider silk and collagen to make a strong, lightweight but tough material. Also use design cues you can extract from leaves.",
           "Now, develop an image generation prompt that I can use with DALLE-3 or Stable Diffusion. ",
           ]

# Create the output directory if it doesn't exist
output_dir = "image_gen_output_V3"
os.makedirs(output_dir, exist_ok=True)

# Iterate through each model
for model_path, model_name in model_associations.items():
    gc.collect()
    try:
        del model
        del tokenizer
    except:
        print ()

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    
    if any(substring in model_name.lower() for substring in ['mistral', 'smollm']):
        eos_token_id = eos_token_id_Mistral   
        #print ("Mistral or SmoLM model detected.")
    else:
        eos_token_id = eos_token_id_Llama      
    print (f"***{model_name}***")
                                     
    # Load the model
    model, tokenizer = load_model(model_path)
    
    # Initialize conversation
    messages = []
    
    # Generate responses for each question
    for question in questions:
        result, messages = generate_response(
            model, tokenizer, text_input=question, num_return_sequences=1, eos_token_id=eos_token_id,
            system_prompt=system_prompt, top_k=512, top_p=0.9, repetition_penalty=1.1,
            temperature=0.2, max_new_tokens=1024, messages=messages, verbatim=False
        )

        # Print the conversation
        print(64 * "-")
        print("User: ", question)
        print(64 * "-")
        print("Assistant: ", result[0])

        # Append assistant's response to the conversation
        messages.append({"role": "assistant", "content": result[0]})
    
    # Save the conversation as a markup file in the output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    filename = f"{model_name}_{timestamp}.md"
    filename = sanitize_filename(filename)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        f.write(f"# Conversation with {model_name}\n\n")
        for message in messages:
            f.write(f"**{message['role'].capitalize()}**: {message['content']}\n\n")

    print(f"Conversation saved to {filepath}")
```

Helper functions to convert .MD markup files to LaTeX text boxes

```python
def convert_md_to_latex(md_file_path, model_name):
    # Read the markdown file
    with open(md_file_path, 'r') as f:
        md_content = f.readlines()
    
    # Remove the line that starts with "Conversation generated by the model:"
    md_content = [line for line in md_content if not line.startswith("# Conversation with")]
    
    # Combine the content back into a single string
    md_content = ''.join(md_content)
    
    model_name = model_name.replace('_', '-')
    
    # Replace the role formatting for System, User, and Assistant with the special lstlisting format
    md_content = md_content.replace("**System**:", r"(*@\bf\hlblue{System:}@*)")
    md_content = md_content.replace("**User**:", r"(*@\bf\hlred{User:}@*)")
    md_content = md_content.replace("**Assistant**:", r"(*@\bf\hllightgreen{Assistant:}@*)")
    
    # Prepare the LaTeX content using lstlisting with the special format
    latex_content = r"""
\begin{figure}[H]
\begin{LLMbox}{""" + 'Conversation with: ' + model_name + r"""}
\begin{lstlisting}
""" + md_content + r"""
\end{lstlisting}
\end{LLMbox}
\caption{Conversation generated by the model: """ + model_name + r""" + '.'}
\end{figure}
"""
    
    # Define the output LaTeX file name and sanitize it
    latex_file_path = md_file_path.replace('.md', '.tex')
   
    # Write the LaTeX content to a new file
    with open(latex_file_path, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX file saved to {latex_file_path}")

# Function to process all .md files in the directory
def process_all_md_files_in_directory(directory, model_associations):
    for model_path, model_name in model_associations.items():
        # Construct the expected .md file name based on the model name
        md_files = glob.glob(os.path.join(directory, f"{sanitize_filename(model_name)}_*.md"))  # Using glob to match files with a pattern
        for md_file_path in md_files:
            convert_md_to_latex(md_file_path, model_name)
```

Example use to convert .MD files in ```output_dir``` to LaTeX code:
```python
output_dir = "text_output"
process_all_md_files_in_directory(output_dir, model_associations)
```

## Image generation using ```leaf-FLUX.1-dev```

You will need significant GPU resources to run FLUX. 

```python
from diffusers import FluxPipeline
import torch

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
repo_id='lamm-mit/leaf-FLUX.1-dev'
pipeline.load_lora_weights(repo_id, )
pipeline=pipeline.to('cuda')
```

Alternatively:
```python
from diffusers import FluxPipeline
import torch

model_id='lamm-mit/leaf-FLUX.1-dev'
pipeline = FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)
pipeline=pipeline.to('cuda')
```

### Sample inference
Helper functions
```
import gc
from datetime import datetime

from tqdm.auto import tqdm

import argparse
import glob
import hashlib

import pandas as pd
import torch
from transformers import T5EncoderModel
from pathlib import Path
 
from diffusers import StableDiffusion3Pipeline

from PIL import Image
def generate_filename(base_name, extension=".png"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"

def save_image(image, directory, base_name="image_grid"):
    
    filename = generate_filename(base_name)
    file_path = os.path.join(directory, filename)
    image.save(file_path)
    print(f"Image saved as {file_path}")

def image_grid(imgs, rows, cols, save=True, save_dir='generated_images', base_name="image_grid",
              save_individual_files=False):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
        if save_individual_files:
            save_image(img, save_dir, base_name=base_name+f'_{i}-of-{len(imgs)}_')
            
    if save and save_dir:
        save_image(grid, save_dir, base_name)
    
    return grid

def download_image(url):
  try:
    response = requests.get(url)
  except:
    return None
  return Image.open(BytesIO(response.content)).convert("RGB")
```
Then, do inference as such: 
```python
prompt="""Generate a futuristic, eco-friendly architectural concept utilizing a biomimetic composite material that integrates the structural efficiency of spider silk with the adaptive porosity of plant tissues. Utilize the following key features:

* Fibrous architecture inspired by spider silk, represented by sinuous lines and curved forms.
* Interconnected, spherical nodes reminiscent of plant cell walls, emphasizing growth and adaptation.
* Open cellular structures echoing the permeable nature of plant leaves, suggesting dynamic exchanges and self-regulation capabilities.
* Gradations of opacity and transparency inspired by the varying densities found in plant tissues, highlighting functional differentiation and multi-functionality.
"""

num_samples = 1
num_rows    = 1
n_steps     = 25
guidance_scale=3.5
all_images = []
for _ in range(num_rows):
    image = pipeline(prompt,num_inference_steps=n_steps,num_images_per_prompt=num_samples,
                     guidance_scale=guidance_scale,).images     
    all_images.extend(image)

grid = image_grid(all_images, num_rows, num_samples,  save_individual_files=True,)
grid
```
Resulting image: 
![image_grid_1-of-25__20240831_185535](https://github.com/user-attachments/assets/3928bef5-aa47-40ee-a296-674547915b37)
