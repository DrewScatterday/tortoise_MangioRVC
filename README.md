<div align="center">
<h1>Fast TorToiSe TTS + Mangio-RVC-Fork</h1>
This repo acts as a pythonic bridge between <a href="https://github.com/152334H/tortoise-tts-fast">tortoise-tts-fast</a> 
and <a href="https://github.com/Mangio621/Mangio-RVC-For">Mangio-RVC-Fork</a><br><br>
    
[![madewithlove](https://github.com/DrewScatterday/tortoise_MangioRVC/blob/main/assets/madewithlove.svg)](https://github.com/DrewScatterday/tortoise_MangioRVC)
[![Licence](https://github.com/DrewScatterday/tortoise_MangioRVC/blob/main/assets/license.svg)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)

<img src="https://github.com/DrewScatterday/tortoise_MangioRVC/blob/main/assets/tortoise.png" width="500" height="500"/><br>
</div>

## Summary: 
A few months ago, I made a fun TTS side project with 11Labs but was frustrated with API costs. I set out to find the best local TTS workflow. After testing I think this is the best (with respect to speed and quality) local TTS option as of September 2023. 

> ⭐ If you like this repo, give it a star and support the projects it's built upon. I'm merely standing on the shoulders of giants.

## High Level Workflow: 
- Use [ai-voice-cloning](https://git.ecker.tech/mrq/ai-voice-cloning) to finetune a tortoise voice model
- Install [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) to train an RVC voice model or use the AI Hub [discord](https://discord.com/invite/aihub) to download a trained voice model file
- Follow the install guide below to clone this repo
- Run `pipeline.py` This pipeline uses fast tortoise, deepspeed, and low quality parameters to get the fastest inference times possible.
- The pipeline shoves the tortoise output into Mangio-RVC to greatly increase the quality of the voice

## Usage: 
Usage is with `pipeline.py`
### Tortoise usage
For Tortoise, import the API, initalize your model, load your voice latents file, set your parameters, and call the `tts_with_preset` function. For more info on all the parameters you can use with these functions, checkout `api.py` in the tortoise repo:
```python
from tortoise.utils.audio import load_audio, load_voice, load_voices
from tortoise.api import TextToSpeech

tts = TextToSpeech(kv_cache=True, use_deepspeed=True, ar_checkpoint="tortoise-tts-fast/Duke.pth")

text = "Hey dude. Thanks for checking out my repo. Be sure to hit that star button. There are some things that are a little hacky. If you make any improvements, open a pull request you sexy son of a gun."
preset = "ultra_fast"
voice = 'duke'
save_tortoise_out = True

vs, conditioning_latents = load_voice(voice)
gen = tts.tts_with_preset(text, voice_samples=None, conditioning_latents=conditioning_latents, preset=preset, num_autoregressive_samples=1, diffusion_iterations=10, cond_free=True, temperature=0.8, half=False)

if save_tortoise_out:
    torchaudio.save('generated.wav', gen.squeeze(0).cpu(), 24000)

gen_resampled = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(gen)
tortoise_out = gen_resampled.squeeze(0).detach().cpu().numpy().flatten()
```

https://github.com/DrewScatterday/tortoise_MangioRVC/assets/28267620/6e230e57-5f40-4efc-bd77-102191fa8933

### Mangio RVC usage: 
For RVC, we'll import the RVC api, create our model and voice params, then call the `vc_single` function to convert the audio: 
```python
from rvc_infer import get_vc, vc_single

# Init model params:
model_path = "weights/DukeNukem.pth"
device="cuda:0"
is_half=False
get_vc(model_path, device, is_half)

# Voice and audio params: 
speaker_id = 0
input_audio = tortoise_out
f0up_key = -2
f0_file = None
f0_method = "rmvpe"
index_path = "logs/added_IVF601_Flat_nprobe_6.index"
index_rate = 0.75
filter_radius = 3
resample_sr = 48000
rms_mix_rate = 0.25
protect = 0.33
crepe_hop_length = 160
wav_opt = vc_single(sid=speaker_id, input_audio=input_audio, f0_up_key=f0up_key, f0_file=f0_file, f0_method=f0_method, file_index=index_path, index_rate=index_rate, filter_radius=filter_radius, resample_sr=resample_sr, rms_mix_rate=rms_mix_rate, protect=protect, crepe_hop_length=crepe_hop_length)

output_audio_path = os.path.join(os.pardir, "test.wav")
wavfile.write(output_audio_path, resample_sr, wav_opt)
```
The whole pipeline only took about 9 seconds on my 3070TI with 8GB VRAM (and thats with the added time of initalizing models and frameworks, you could get even faster if you ran this like a WebGUI like other frameworks do), not bad at all. 

https://github.com/DrewScatterday/tortoise_MangioRVC/assets/28267620/9c9852ce-90fc-445e-a89e-96f30e9d2a6b


## Installation: 
As a disclaimer, installing this is not simple and quite hacky. You will need to do the following: 
- Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- Install [ai-voice-cloning](https://git.ecker.tech/mrq/ai-voice-cloning) to create finetuned tortoise models. Here's a video [guide](https://youtu.be/6sTsqSQYIzs?si=dva0uYGnKwxpQJg2)
- Install Mangio RVC Fork, you can do this by cloning the repo or using 7zip. I recommend 7zip as its much easier [Mangio RVC 7zip install guide](https://docs.google.com/document/d/1KKKE7hoyGXMw-Lg0JWx16R8xz3OfxADjwEYJTqzDO1k/edit) (if this is out of date check the AI hub discord for up to date installation)
- Once you have these installed do `git clone https://github.com/DrewScatterday/tortoise_MangioRVC.git`
- Once cloned, make sure the RVC Mangio Fork folder is placed within this directory
- Next clone fast tortoise. I would recommend using my fork as it has deepspeed implemented for maximum speed. But you can also use the original `git clone https://github.com/152334H/tortoise-tts-fast.git`
- Then do the following commands:
```
conda create --name tortoiseRVC python=3.9 numba inflect
conda activate tortoiseRVC
conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install transformers=4.29.2
pip install -r requirements.txt 
pip3 install git+https://github.com/152334H/BigVGAN.git
python diditwork.py
```
- `diditwork.py` will run the whole pipeline with default models and parameters to make sure its working. If it worked you will see a file called `itworked.wav`

## Other helpful resources:
- [Tortoise + RVC](https://www.youtube.com/watch?v=IcpRfHod1ic) (this was the inspiration for the creation of this repo, thanks Jarrod!)
- The best [settings](https://www.youtube.com/watch?v=fYEdKwqwiG4) for speed and quality for tortoise inference 
- NanoNomad on [speeding up tortoise](https://www.youtube.com/watch?v=Fzah3eJabOY) and [finetuning tortoise](https://www.youtube.com/watch?v=P3BbCG0hTwU)

## Disclaimers: 
- This repo is purely for fun. It has no association with my employer and only my personal hardware was used in the creation of this repo.
- This repo is open source, there will be bugs and it is very much a work in progress. 
- There are ethical concerns with this technology. Here is a link to the original repo discussing [concerns](https://github.com/neonbjb/tortoise-tts#ethical-considerations). I've mostly been using it for silly jokes and to have fun. I'm not responsible for actions that come from this repo, check out the license for more details. Please be a good human being :)
- Lastly, this repo is aimed at being a python API/bridge between these two tools. If you are after a GUI implementation I would recommend this [repo](https://github.com/rsxdalv/tts-generation-webui) or this [repo](https://github.com/litagin02/rvc-tts-webui) (although I don't think it will be as fast or high quality as this repo) 

## Resources and Licenses: 
- [bark-with-voice-clone](https://github.com/serp-ai/bark-with-voice-clone) - [MIT License](https://github.com/serp-ai/bark-with-voice-clone/blob/main/LICENSE.md) 
- [rvc-tts-pipeline](https://github.com/JarodMica/rvc-tts-pipeline) - [MIT License](https://github.com/JarodMica/rvc-tts-pipeline/blob/master/LICENSE)
- [tortoise-tts](https://github.com/neonbjb/tortoise-tts) - [Apache 2.0 License](https://github.com/neonbjb/tortoise-tts/blob/main/LICENSE)
- [tortoise-tts-fast](https://github.com/152334H/tortoise-tts-fast) - [APGL License](https://github.com/152334H/tortoise-tts-fast/blob/main/LICENSE)
- [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) - [MIT License](https://github.com/Mangio621/Mangio-RVC-Fork/blob/main/LICENSE)
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - [MIT License](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)

## Improvements:  
- [ ] Adopt Streamlit UI from fast tortoise fork for easier use and even faster inference times (since models can be loaded into seperate from inference)
- [ ] Dockerfile that will run the UI straight away for easier usage 
- [ ] Make install process less hacky with a .bat setup file or having a .7z file that has everything installed
- [ ] Maybe create a precompiled PYPI package that makes it easier to use
- [ ] Do some testing on a 3090/4090 to get some speed benchmarks 

