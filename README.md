# GenDrum: Using Convolution Neural Networks to Convert Clapping into Drums 
### Abstract 
Numerous text-to-music models allow for music generation using descriptions. Some models also allow for audio input to edit songs based on text prompts. However, most models don’t allow for precise control over individual parts of a song, especially drums or other percussive instruments. To solve this problem, we used Convolutional Neural Networks (CNNs) to classify audio samples of claps, snaps and other percussive hits. The classifications and hit timings are then used to generate a cohesive drumbeat, allowing for direct control of rhythm while generating a unique drumbeat.

View full paper here: [paper](https://1drv.ms/w/c/a9be6d79a6c9ee92/IQAhAHkvAkbITKD2uaxTojFRAWxfIAnGqLUybSM9lRdY6Zk?e=4gFZQr)
View demo here: [demo](https://youtu.be/QR2ianZeS9Q)
## Getting Started
- Install Python 3.11 (I ran into issues running on later python version so this is the only one I can guarrantee works)
- run `pip install`
- record a wav file of yourself clapping/playing a drumbeat out on your desk. Make sure the audio is clear and loud, but isn't clipping.
- open `live.py` and edit the filename variable to point to wherever your audio file is saved. I recommend saving in the input folder.
- run `live.py`. The outputted audio will be saved at `output/beat.wav`.
  
