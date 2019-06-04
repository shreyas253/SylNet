# SylNet

INTRODUCTION
------------
Tensorflow (Python) implementation of a Syllable Count estimator using gated Convolutional Neural Network (CNN) model with Gated activations, Residual connections, dilations and PostNets and an LSTM.

Comments/questions are welcome! Please contact: shreyas.seshadri@aalto.fi

Last updated: 03.06.2019


LICENSE
-------

Copyright (C) 2019 Shreyas Seshadri, Aalto University

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The source code must be referenced when used in a published work.

FILES AND FUNCTIONS
-------------------
train_SylNet.py - Train main SylNet model (pre-trained model avaliable in ./trained_models/). Files ./config_files/config_files.txt and ./config_files/config_sylls.txt contain the paths to .wav sound files and number of syllables for each .wav file respectively.

adap_SylNet.py - Adapt main SylNet model. Files ./config_files/config_files_adap.txt and ./config_files/config_sylls_adap.txt contain the paths to .wav sound files and number of syllables for each .wav file respectively.

validate_SylNet.py	- Test trained model. Files ./config_files/config_files_test.txt and ./config_files/config_sylls_test.txt contain the paths to .wav sound files and number of syllables for each .wav file respectively.

run_SylNet.py	- Run trained model to get sylable counts. Files ./config_files/config_files_run.txt contain the paths to .wav sound files. The files results.txt will contain predicted the syllable counts for the .wav files in ./config_files/config_files_run.txt.

REQUIRED PACKAGES
-------------------
- LibRosa
- TensorFlow
- SciPy
- Numpy

REFERENCES
---------
[1] Seshadri S. & Räsänen O. SylNet: An Adaptable End-to-End Syllable Count Estimator for Speech. Submitted to a Journal.
