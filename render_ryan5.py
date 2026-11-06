import asyncio, base64, json, math, re, subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output"
OUT.mkdir(exist_ok=True)
SR=48000

def decrypt_text():
    payload=json.loads((ROOT/"payload.json").read_text())
    parts=[]
    for i,c in enumerate(payload["chunks"]):
        enc=ROOT/".secure"/f"enc_{i}.bin"
        dec=ROOT/".secure"/f"dec_{i}.txt"
        enc.write_bytes(base64.b64decode(c))
        subprocess.run([
            "openssl","pkeyutl","-decrypt",
            "-inkey",str(ROOT/".secure/private.pem"),
            "-pkeyopt","rsa_padding_mode:oaep",
            "-pkeyopt","rsa_oaep_md:sha256",
            "-pkeyopt","rsa_mgf1_md:sha256",
            "-in",str(enc),"-out",str(dec)
        ],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        parts.append(dec.read_bytes())
    lines=[x.strip() for x in b"".join(parts).decode("utf-8").split("\n\n") if x.strip()]
    if len(lines)!=5: raise RuntimeError(f"Expected 5 affirmations, got {len(lines)}")
    return lines

def canon(s): return re.sub(r"[^a-z0-9]+","",s.lower())

PROSODY=[
"You are powerful as fuck... to fight against the matrix, and beat it, and rule over it.",
"You are a high value male, worthy of respect and ass kissing... and if not, you can put them in place.",
"You are a sexually attractive Chad, who bang Queens, and give them epitome of sexual pleasure.",
"You are rich as fuck... and wipe ass with 1000 Indian Rupee Notes.",
"You are an alpha sigma hybrid male."
]

def filt(x,hz,kind,order=2):
    sos=butter(order,hz/(SR/2),btype=kind,output="sos")
    return sosfilt(sos,x,axis=0)

async def synth_one(text,rate,out):
    import edge_tts
    c=edge_tts.Communicate(
        text=text, voice="en-GB-RyanNeural",
        rate=rate, pitch="-2Hz",
        connect_timeout=15, receive_timeout=30
    )
    await asyncio.wait_for(c.save(str(out)),timeout=45)

def trim_mp3(src,dst):
    subprocess.run([
        "ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(src),
        "-af","silenceremove=start_periods=1:start_duration=0.03:start_threshold=-52dB:stop_periods=-1:stop_duration=0.12:stop_threshold=-52dB",
        "-ar","48000","-ac","1","-c:a","pcm_f32le",str(dst)
    ],check=True)

def read_mono(path):
    x,sr=sf.read(path,dtype="float32")
    if x.ndim>1:x=x.mean(axis=1)
    if sr!=SR: raise RuntimeError("Unexpected sample rate")
    return x

def roomtone(seconds,seed):
    n=int(seconds*SR)
    rng=np.random.default_rng(seed)
    z=rng.normal(0,1,n).astype(np.float32)
    z=filt(z,120,"highpass"); z=filt(z,9000,"lowpass")
    z/=np.sqrt(np.mean(z*z)+1e-12)
    return z*(10**(-68/20))

async def synth_lines(lines):
    for a,b in zip(lines,PROSODY):
        if canon(a)!=canon(b): raise RuntimeError("Prosody text changed spoken words")
    rates=["-9%","-6%","-9%","-7%","-11%"]
    rawdir=OUT/"segments"; rawdir.mkdir(exist_ok=True)
    parts=[]
    for i,(t,r) in enumerate(zip(PROSODY,rates)):
        mp3=rawdir/f"{i}.mp3"; wav=rawdir/f"{i}.wav"
        await synth_one(t,r,mp3)
        trim_mp3(mp3,wav)
        parts.append(read_mono(wav))
    return parts

def assemble(parts):
    gaps=[0.72,0.82,0.76,0.90]
    chunks=[]
    gains=[1.00,0.97,0.99,1.02,1.05]
    for i,x in enumerate(parts):
        chunks.append(x*gains[i])
        if i<4: chunks.append(roomtone(gaps[i],100+i))
    mono=np.concatenate(chunks)
    d=int(.008*SR); delayed=np.pad(mono,(d,0))[:len(mono)]
    return np.column_stack([mono,.985*mono+.015*delayed]).astype(np.float32)

def voice_dsp(v):
    v=filt(v,72,"highpass"); v=filt(v,15500,"lowpass")
    thr=10**(-18/20); mag=np.abs(v); over=np.maximum(mag-thr,0)
    comp=np.sign(v)*(np.minimum(mag,thr)+over/1.55)
    return (.90*v+.10*comp).astype(np.float32)

def osc(f,t,p=0): return np.sin(2*np.pi*f*t+p).astype(np.float32)

def music_bed(n):
    t=np.arange(n,dtype=np.float32)/SR
    L=np.zeros(n,np.float32);R=np.zeros(n,np.float32)
    prog=[[73.42,87.31,110],[65.41,82.41,98],[69.3,87.31,103.83],[61.74,73.42,92.5],[55,69.3,82.41]]
    seg=7.2
    for j,freqs in enumerate(prog):
        s=int(j*seg*SR);e=min(n,s+int((seg+2.2)*SR))
        if s>=n:break
        tt=np.arange(e-s,dtype=np.float32)/SR
        env=(np.sin(np.linspace(0,np.pi,e-s))**0.58).astype(np.float32)
        a=np.zeros(e-s,np.float32);b=np.zeros(e-s,np.float32)
        for k,f in enumerate(freqs):
            a+=osc(f,tt,.4*j+.6*k)+.16*osc(2*f,tt,.8+k)
            b+=osc(f,tt,.4*j+.6*k+.14)+.16*osc(2*f,tt,1.0+k)
        a/=len(freqs)*1.2;b/=len(freqs)*1.2
        L[s:e]+=.032*env*a;R[s:e]+=.032*env*b
    sub=.0055*osc(36.7,t,.2);L+=sub;R+=sub
    rng=np.random.default_rng(777)
    air=rng.normal(0,1,n).astype(np.float32)
    air=filt(air,6000,"highpass");air=filt(air,14000,"lowpass")
    air/=np.sqrt(np.mean(air*air)+1e-12);air*=.00125
    L+=air;R+=np.roll(air,int(.022*SR))*.96
    return filt(filt(np.column_stack([L,R]),28,"highpass"),14500,"lowpass")

def envelope(v):
    mono=np.mean(np.abs(v),axis=1);w=int(.035*SR)
    e=np.convolve(mono,np.ones(w,dtype=np.float32)/w,mode="same")
    out=np.empty_like(e);y=0.0
    au=math.exp(-1/(SR*.018));ad=math.exp(-1/(SR*.34))
    for i,z in enumerate(e):
        a=au if z>y else ad;y=a*y+(1-a)*z;out[i]=y
    return out

def loudnorm(src,wav_out,mp3_out,target=-15.0):
    p=subprocess.run(["ffmpeg","-hide_banner","-nostats","-i",str(src),"-af",f"loudnorm=I={target}:TP=-1.2:LRA=7:print_format=json","-f","null","-"],capture_output=True,text=True,check=True)
    m=re.search(r'\{\s*"input_i".*?\}',p.stderr,re.S);d=json.loads(m.group(0))
    flt=(f"loudnorm=I={target}:TP=-1.2:LRA=7:measured_I={d['input_i']}:measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:offset={d['target_offset']}:linear=true")
    subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(src),"-af",flt,"-ar","48000","-ac","2","-c:a","pcm_s24le",str(wav_out)],check=True)
    subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(wav_out),"-c:a","libmp3lame","-b:a","320k",str(mp3_out)],check=True)

async def main():
    lines=decrypt_text();parts=await synth_lines(lines);voice=voice_dsp(assemble(parts))
    clean_pre=OUT/"clean_pre.wav";sf.write(clean_pre,voice,SR,subtype="PCM_24")
    loudnorm(clean_pre,OUT/"Ryan_5_Affirmations_Clean.wav",OUT/"Ryan_5_Affirmations_Clean.mp3",-15.5)
    intro=int(2.2*SR);outro=int(4.2*SR)
    fv=np.zeros((intro+len(voice)+outro,2),np.float32);fv[intro:intro+len(voice)]=voice
    bed=music_bed(len(fv));act=np.clip((envelope(fv)-.0025)/.04,0,1);bed*=10**((-5.0*act)/20)[:,None]
    mix=fv+.70*bed;mix=np.tanh(mix*1.045)/np.tanh(1.045)
    fi=int(1.0*SR);fo=int(3.2*SR);mix[:fi]*=np.linspace(0,1,fi)[:,None];mix[-fo:]*=np.linspace(1,0,fo)[:,None]
    cin_pre=OUT/"cin_pre.wav";sf.write(cin_pre,mix,SR,subtype="PCM_24")
    loudnorm(cin_pre,OUT/"Ryan_5_Affirmations_Cinematic.wav",OUT/"Ryan_5_Affirmations_Cinematic.mp3",-15.0)
    (OUT/"qc.json").write_text(json.dumps({"voice":"en-GB-RyanNeural","affirmations":5},indent=2))

if __name__=="__main__": asyncio.run(main())
