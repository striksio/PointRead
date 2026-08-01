# PointRead

Point at word / sentance in a paper book to get the translation. A computer vision pipeline that runs on Nvidia Jetson Orin Nano. Tracks the finger over printed text, after activation with double pinch, reads it and translates through speach. 

> Status: Early, started project.

## Problem

Reading books with unknown words is problematic, you have to go to a dictionary or translation. While on digital devices, such as phones, in-built translation is easy to access by just clicking on the word / selecting the text and specifying translation. But on paper books, which many people prefer, there are few options. First, to translate on the digital devices by inputting it, which although works, wants to be avoided due to the screen's effects on eyes. Second, is to actually open a dictionary book and look for words, which is fair, but takes time, and slows down reading.

## How it works

Index finger and thumbs up work for activation, which is a double pinch. Track the index finger and the text above it. Once done, track the text that was pointed at. Extract, recognise, translate, speak.

## Hardware

Nvidia Jetson Orin Nano and Raspberry Pi Camera Module V2 (8MP Sony IMX219 Sensor). 

## Roadmap
 
- [x] Real-time hand detection and 21-point landmark tracking on Jetson
- [x] Index finger and thumb tracking with a pinch gesture
- [ ] Text region extraction from the pointed span
- [ ] OCR of the selected text
- [ ] Translation
- [ ] Text to speech through a speaker
- [ ] Move off the browser preview toward a headless, audio-only device
