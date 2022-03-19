# Based on works by:
# TensorFlow Lite Tutorial Part 3: Speech Recognition on Raspberry Pi By ShawnHymel
# https://www.digikey.com/en/maker/projects/tensorflow-lite-tutorial-part-3-speech-recognition-on-raspberry-pi/8a2dc7d8a9a947b4a953d37d3b271c71
# 
# Aaron Judy
# https://theneverworks.com/
# https://followalongwith.us/
# 
# Mozilla
# https://github.com/mozilla/DeepSpeech
#
# Zero Iron Man Suit AI Assistant v0.4

import sounddevice as sd
import numpy as np
import scipy.signal
import os
import operator
import marshal
import aiml
import csv
import urllib.request  as urllib2
import json
import timeit
import re
import python_speech_features
import RPi.GPIO as GPIO
from scipy.io.wavfile import write
from tflite_runtime.interpreter import Interpreter
import  soundfile as sf
import glob
import sys
import platform
import time
import nltk
from nltk.tokenize import word_tokenize
import serial
import deepspeech
import wave

# Parameters
debug_time = 0
debug_acc = 0
led_pin = 8
suit_pin = 11
face_pin = 13
reactor_pin = 15
word_threshold = 0.7
rec_duration = 0.5
cmd_duration = 3
window_stride = 0.5
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16
model_path = '/home/pi/ai/wake_word_zero_lite.tflite'
voxactivated = False
model_file_path = 'deepspeech-0.9.3-models.tflite'
model = deepspeech.Model(model_file_path)
scorer_file_path = 'deepspeech-0.9.3-models.scorer'
model.enableExternalScorer(scorer_file_path)
lm_alpha = 0.75
lm_beta = 1.85
model.setScorerAlphaBeta(lm_alpha, lm_beta)
beam_width = 500
model.setBeamWidth(beam_width)

connecteddevicecount = 0
try:
    ser0 = serial.Serial('/dev/ttyACM0', 9600)
    connecteddevicecount += 1
except:
    ser0 = None
try:
    ser1 = serial.Serial('/dev/ttyACM1', 9600)
    connecteddevicecount += 1
except:
    ser1 = None

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

# GPIO - Unused currently, available if no Arduino
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#print(input_details)

EXAMPLE_COMMAND = "do"

# AIML Directory
saiml = "/home/pi/ai/aiml/"

# brain
k = aiml.Kernel()

# setpreds() function
def setpreds():
    with open(saiml + 'preds.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            #print((row[0]), (row[1]))
            k.setBotPredicate((row[0]), (row[1]))
    plat = platform.machine()
    osys = os.name
    print("Zero for " + osys)
    print("System Architecture " + plat)
    #print "Memory " + psutil.virtual_memory()
    k.setBotPredicate("architecture", plat)
    k.setBotPredicate("os", osys)


# get_oldest_file() function
def get_oldest_file(files, _invert=False):
    """ Find and return the oldest file of input file names.
    Only one wins tie. Values based on time distance from present.
    Use of `_invert` inverts logic to make this a youngest routine,
    to be used more clearly via `get_youngest_file`.
    """
    gt = operator.lt if _invert else operator.gt
    # Check for empty list.
    if not files:
        return None
    # Raw epoch distance.
    now = time.time()
    # Select first as arbitrary sentinel file, storing name and age.
    oldest = files[0], now - os.path.getmtime(files[0])
    # Iterate over all remaining files.
    for f in files[1:]:
        age = now - os.path.getmtime(f)
        if gt(age, oldest[1]):
            # Set new oldest.
            oldest = f, age
    # Return just the name of oldest file.
    return oldest[0]


# get_youngest_file() function
def get_youngest_file(files):
    return get_oldest_file(files, _invert=True)


# learn() function
def learn(aimlfiles):
    if not aimlfiles:
        k.learn(saiml + "xfind.aiml")
    for f in aimlfiles[1:]:
        k.learn(f)


# brain() function
def brain():
    if os.path.isfile(saiml + "zero.brn"):
        brnfiles = glob.glob(saiml + "*.brn")
        aimlfiles = glob.glob(saiml + "*.aiml")
        brn = get_oldest_file(brnfiles)
        oldage = os.path.getmtime(brn)
        youngaiml = get_youngest_file(aimlfiles)
        youngage = os.path.getmtime(youngaiml)
        testfiles = [brn, youngaiml]
        oldfile = get_oldest_file(testfiles)
        if (oldfile != (saiml + "zero.brn")):
            k.bootstrap(brainFile=saiml + "zero.brn")
        else:
            learn(aimlfiles)
    else:
        aimlfiles = glob.glob(saiml + "*.aiml")
        learn(aimlfiles)
        setpreds()
    if os.path.isfile(saiml + "zero.ses"):
        sessionFile = file(saiml + "zero.ses", "rb")
        session = marshal.load(sessionFile)
        sessionFile.close()
        for pred,value in session.items():
            k.setPredicate(pred, value, "zero")
    else:
        setpreds()
    k.saveBrain(saiml + "zero.brn")

def handle_command(command):
    sent = nltk.word_tokenize(command.lower())
    if ("turn" in sent and "lights" in sent and "on" in sent):
        if ser0 != None:
            ser0.write(b'B')
            response = "Suit lights on"
        else:
            response = "That Arduino is not connected"
    elif ("turn" in sent and "lights" in sent and "off" in sent):
        if ser0 != None:
            ser0.write(b'C')
            response = "Suit lights off"
        else:
            response = "That Arduino is not connected"
    elif ("laser" in sent and "off" in sent):
        if ser0 != None:
            ser0.write(b'D')
            response = "Laser off"
        else:
            response = "That Arduino is not connected"
    elif ("laser" in sent and "on" in sent):
        if ser0 != None:
            ser0.write(b'L')
            response = "Firing laser"
        else:
            response = "That Arduino is not connected"
    elif ("my" in sent and "turn" in sent):
        if ser0 != None:
            ser0.write(b'M')
            response = "Armed and ready"
        else:
            response = "That Arduino is not connected"
    elif ("stand" in sent and "down" in sent):
        if ser0 != None:
            ser0.write(b'S')
            response = "Standing down"
        else:
            response = "That Arduino is not connected"
    elif ("mask" in sent and "open" in sent):
        if ser1 != None:
            ser1.write(b'O')
            response = "Faceplate open"
        else:
            response = "That Arduino is not connected"
    elif ("mask" in sent and "closed" in sent):
        if ser1 != None:
            ser1.write(b'C')
            response = "Faceplate closed"
        else:
            response = "That Arduino is not connected"
    else:
        response = k.respond(command)
    return response

def agentsay(message):
    try:
        send = message
        dothisnow = 'swift -n Callie -o /home/pi/tmp.wav "'+send+'" && aplay /home/pi/tmp.wav'
        print(dothisnow)
        os.system(dothisnow)
        return
    except:
        print("Unable to speak")
        response = "Unable to speak"
        return response

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs

    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):
    global voxactivated

    GPIO.output(led_pin, GPIO.LOW)

    # Start timing for testing
    start = timeit.default_timer()

    # Notify if errors
    if status:
        print('Error:', status)

    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)

    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)

    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Compute features
    mfccs = python_speech_features.base.mfcc(window, 
                                        samplerate=new_fs,
                                        winlen=0.256,
                                        winstep=0.050,
                                        numcep=num_mfcc,
                                        nfilt=26,
                                        nfft=2048,
                                        preemph=0.0,
                                        ceplifter=0,
                                        appendEnergy=False,
                                        winfunc=np.hanning)
    mfccs = mfccs.transpose()

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0][0]
    if val > word_threshold:
        print('Keyword detected')
        print(val)
        GPIO.output(led_pin, GPIO.HIGH)
        voxactivated = True

    if debug_acc:
        print(val)

    if debug_time:
        print(timeit.default_timer() - start)

brain()
agentsay("I am online and loaded. I detected " + str(connecteddevicecount) + " connected Arduinos.")
while True:
    # Start streaming from microphone
    with sd.InputStream(channels=num_channels,
                        samplerate=sample_rate,
                        blocksize=int(sample_rate * rec_duration),
                        callback=sd_callback) as xtream:
        while voxactivated == False:
            pass
    sd.wait()
    sd.stop(ignore_errors=True)
    fs = 16000  # Sample rate
    seconds = 3  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    write('/home/pi/output.wav', fs, myrecording)
    sd.stop(ignore_errors=True)

    filename = '/home/pi/output.wav'
    w = wave.open(filename, 'r')
    rate = w.getframerate()
    frames = w.getnframes()
    buffer = w.readframes(frames)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    lines = model.stt(data16)
    sayback = handle_command(str(lines))
    agentsay(sayback)
    voxactivated = False # Setting true, loops conversation in near real time
