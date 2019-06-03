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
train_main_model.py - Train main SylNet model (pre-trained model avaliable in ./trained_models/). Change line 20 in the code to your current path. Files config_files.txt and config_sylls.txt contain the paths to .wav sound files and number of syllables for each .wav file respectively.

train_adap_model.py - Adapt main SylNet model. Change line 20 in the code to your current path. Files config_files_adap.txt and config_sylls_adap.txt contain the paths to .wav sound files and number of syllables for each .wav file respectively.

test_model.py	- Test trained model. Change line 20 in the code to your current path. Change line 24 in the code to your choice of the trained model.  Files config_files_test.txt and config_sylls_test.txt contain the paths to .wav sound files and number of syllables for each .wav file respectively.

run_model.py	- Run trained model to get . Change line 20 in the code to your current path. Change line 24 in the code to your choice of the trained model. Files config_files_run.txt contain the paths to .wav sound files. The files results.npy will contain predicted the syllable counts for the .wav files in config_files_run.txt.



REFERENCES
---------
[1] Seshadri S. & Räsänen O. SylNet: An Adaptable End-to-End Syllable Count Estimator for Speech. Submitted to a Journal.
