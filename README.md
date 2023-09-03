<div align="center">

<h1>Fast TorToiSe TTS + Mangio-RVC-Fork </h1>

This repo acts as a pythonic bridge between <a href="https://github.com/152334H/tortoise-tts-fast">tortoise-tts-fast</a> 
and <a href="https://github.com/Mangio621/Mangio-RVC-For">Mangio-RVC-Fork</a><br><br>

[![madewithlove](https://github.com/DrewScatterday/tortoise_MangioRVC/blob/main/assets/madewithlove.svg)](https://github.com/DrewScatterday/tortoise_MangioRVC)
[![Licence](https://github.com/DrewScatterday/tortoise_MangioRVC/blob/main/assets/license.svg)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)

<img src="https://github.com/DrewScatterday/tortoise_MangioRVC/blob/main/assets/tortoise.png" width="500" height="500"/><br>
</div>

# Summary: 
A few months ago, I made a fun TTS side project. I initially used ElevenLabs but was frustrated with the API costs I was paying. I wanted to find the best self hosted TTS workflow and after doing a lot of testing, I personally think this is the fastest and best sounding local TTS option as of September 2023. 

The high level workflow is to use a finetuned TorToiSe model with low quality inference parameters to capture porosity and tone of a voice quickly. After a lower audio quality tortoise output is produced, it is shoved into Mangio-RVC to increase the quality and pitch of a the voice sample. This pipeline uses tortoise-tts-fast with deepspeed and low quaslity inference parameters to get the fastest inference times possible.

> If you enjoy this repo, please support me by dropping a star on it. And please also support the other projects that this is built upon. I am merely standing on the shoulders of giants. 

# Usage: 
Usage is simple and requires editing `pipeline.py`
### Tortoise usage
First, import the tortoise API and initalize your model:
```python
from tortoise.utils.audio import load_audio, load_voice, load_voices
from tortoise.api import TextToSpeech

tts = TextToSpeech(kv_cache=True, use_deepspeed=True, ar_checkpoint="tortoise-tts-fast/Duke.pth")
```
Next, set up your fintuned model checkpoint, model parameters, and voice latents file: 
```python
text = "Hey dude, thanks for checking out my repo. Be sure to hit that star button. There are some things that are a little hacky. If you make any improvements, open a pull request you sexy son of a gun."
preset = "ultra_fast"
voice = 'duke'
vs, conditioning_latents = load_voice(voice)
save_tortoise_out = True
```
And lastly predict with tortoise and convert/flatten the tensor to be shoved into RVC. You can optionally save the tensor to wav file on disk:
```python
gen = tts.tts_with_preset(text, voice_samples=None, conditioning_latents=conditioning_latents, preset=preset, num_autoregressive_samples=1, diffusion_iterations=10, cond_free=True, temperature=0.8, half=False)
if save_tortoise_out:
    torchaudio.save('generated.wav', gen.squeeze(0).cpu(), 24000)
gen_resampled = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(gen)
tortoise_out = gen_resampled.squeeze(0).detach().cpu().numpy().flatten()
```
Here's what the wav audio produced from tortoise sounds like: 
INSERT FILE HERE

### Mangio RVC usage: 
First, we'll import the RVC api and initalize our model params: 
```python
from rvc_infer import get_vc, vc_single

# Init model params:
model_path = "weights/DukeNukem.pth"
device="cuda:0"
is_half=False
get_vc(model_path, device, is_half)
```
Next, we'll set our voice audio parameters and call the vv_single function: 
```python
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
```
And lastly write our wav output to disk: 
```python
output_audio_path = os.path.join(os.pardir, "test.wav")
wavfile.write(output_audio_path, resample_sr, wav_opt)
```
Here's what the final pipeline sounds like, and the whole thing only took about 10 seconds on my 3070TI with 8GB VRAM, not bad. 
INSERT WAV FILE HERE

# Installation: 
As a warning, installtion is not simple. I think this could be improved in the future because right now it is quite hacky. You will need to do the following: 
- Install miniconda
- Install RVC Mangio Fork by following these instructions. If these are out of date, the AI hub discord server will have up to date RVC installtion tutorials. You can also follow the manual install on the Mangio Github fork
- Make sure the RVC Mangio Fork is placed within this directory 

# Dude this README is way too long and I have no idea where to start: 
If you're lost start here.
- Watch Jarrods video on installing mrqs ai voice cloning 
- Once you've got a nice finetuned model that you are happy with, watch this video on installing Mangio RVC.
- Once you've trained a RVC v2 model or downloaded one from the AI hub discord, clone this repo follow the above installation steps.
- Lastly, this repo is aimed at being a nice python API/bridge between this two tools. If you are after a GUI implementation I would recommend this repo here. Or maybe some chad contributor will take the time to add a streamlit UI to this repo.

# Resources and Licenses: 

# Improvements that could be done:  
- [ ] Add Streamlit UI for easier to use
- [ ] Make installing process less hacky with a .bat setup file or having a .7z file that has everything installed
- [ ] Maybe create a precompiled PYPI package that makes it easier to use

