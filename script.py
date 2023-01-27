import os
from pydub import AudioSegment

#à modifier
file = ''
sound = os.getcwd() + r"/data/sound_library/" + file

# Load the .wav file
audio = AudioSegment.from_wav(sound)

# Determine the length of the audio in milliseconds
length = len(audio)

# Set the duration of each part (in milliseconds)
part_duration = 2000 # 60 seconds

# Initialize a list to hold the parts
parts = []

# Split the audio into parts
for i in range(0, length, part_duration):
    part = audio[i:i+part_duration]
    parts.append(part)

# Save each part as a separate .wav file
for i, part in enumerate(parts):
    part.export( os.getcwd() + f"part{i}.wav", format="wav")


