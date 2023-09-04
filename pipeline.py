import torchaudio
import time
import os
import sys
import scipy.io.wavfile as wavfile

now_dir=os.getcwd()
sys.path.append(os.path.join(now_dir, "tortoise-tts-fast"))
from tortoise.utils.audio import load_audio, load_voice, load_voices
from tortoise.api import TextToSpeech

tts = TextToSpeech(kv_cache=True, ar_checkpoint="tortoise-tts-fast/Duke.pth")

text = "Hey dude. Thanks for checking out my repo. Be sure to hit that star button. There are some things that are a little hacky. If you make any improvements, open a pull request you sexy son of a gun."
preset = "ultra_fast"
voice = 'duke'
vs, conditioning_latents = load_voice(voice)
save_tortoise_out = True

t0 = time.time()
gen = tts.tts_with_preset(text, voice_samples=None, conditioning_latents=conditioning_latents, preset=preset, num_autoregressive_samples=1, diffusion_iterations=10, cond_free=True, temperature=0.8, half=False)
t1 = time.time()
total = t1-t0
print("Tortoise took " + str(total) + " seconds")

if save_tortoise_out:
    torchaudio.save('generated.wav', gen.squeeze(0).cpu(), 24000)
gen_resampled = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(gen)
tortoise_out = gen_resampled.squeeze(0).detach().cpu().numpy().flatten()


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

# Write output:
output_audio_path = os.path.join(os.pardir, "test.wav")
wavfile.write(output_audio_path, resample_sr, wav_opt)
t2 = time.time()
total = t2-t0
print("Whole pipeline took " + str(total) + " seconds")
