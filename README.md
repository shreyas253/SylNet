# SylNet

INTRODUCTION
------------
Tensorflow (Python) implementation of a Syllable Count estimator using gated Convolutional Neural Network (CNN) model with Gated activations, Residual connections, dilations and PostNets and an LSTM.

Comments/questions are welcome! Please contact: shreyas.seshadri@aalto.fi

Please cite the following paper whenever using these codes, their derivatives, or another
implementation of SylNet in a publication:

Seshadri S. & Räsänen O. SylNet: An Adaptable End-to-End Syllable Count Estimator for Speech. IEEE Signal Processing Letters, 26, 1359–1363, 2019.


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

FILES AND FUNCTIONS – A QUICK GUIDE:
-------------------
train_SylNet.py - Train main SylNet model (pre-trained model available in ./trained_models/). Files ./config_files/config_files.txt and ./config_files/config_sylls.txt contain the paths to .wav sound files and number of syllables for each .wav file, respectively.

adap_SylNet.py - Adapt main SylNet model. Files ./config_files/config_files_adap.txt and ./config_files/config_sylls_adap.txt contain the paths to .wav sound files and number of syllables for each .wav file, respectively.

validate_SylNet.py	- Test trained model. Files ./config_files/config_files_test.txt and ./config_files/config_sylls_test.txt contain the paths to .wav sound files and number of syllables for each .wav file, respectively.

run_SylNet.py	- Run trained model to get syllable counts. Files ./config_files/config_files_run.txt contain the paths to .wav sound files. The files results.txt will contain predicted the syllable counts for the .wav files in ./config_files/config_files_run.txt.

run_SylNet_adapted.py - Same as above, but uses the adapted model that resulted from adap_SylNet.py


USING THE PRE-TRAINED MODEL TO SYLLABIFY SPEECH DATA
-------------------

SYNTAX:
python run_SylNet.py <input_files> <result_file>

where <input_files> can be:
  1) A path to a single .wav file, e.g.:

      python run_SylNet.py /path_to/audiofile.wav results.txt

  2) A path to a folder with .wav files (to process all the .wavs), e.g.:

      python run_SylNet.py /path_to/my_audiofiles/ results.txt

  3) A path to a .txt file where each row of the text file contains a path to a
      .wav file to be processed, e.g.:

      python run_SylNet.py /path_to/my_filepointers/files_to_process.txt results.txt

By default (without any input arguments), run_SylNet.py will attempt to load files
specified in config_files/config_files_run.txt and stores the results in
results.txt located in the main directory of SylNet.

OUTPUTS:
  Result file (e.g., results.txt) will contain an integer number corresponding to
  the estimated syllable count, one number per row corresponding to each input file
  provided to the algorithm.

  If a folder path with .wavs in it was provided as the input, a separate output file
  of form <resultfilename_without_extension>_files.txt with processed audio filenames  
  will be produced, allowing mapping of output counts to input files.

ADAPTED MODELS:

Use run_SylNet_adapted.py to perform syllable counting with an adapted model.
Otherwise syntax is the same.

NOTE:

The current implementation of SylNet supports maximum output syllable count of 91 syllables per input signal. If your input data is suspected to contain more than 91 syllables (approx. 18 seconds of speech), consider dividing the data into shorter segments before applying SylNet.


REQUIRED PACKAGES
-------------------
- Python 3 (tested on 3.6.5)
- LibROSA (tested with 0.6.3, and 0.7, https://librosa.github.io/librosa/)
- TensorFlow (tested on version 1.10.1). Does NOT currently work on Tensorflow 2.1. 
- SciPy
- Numpy

REFERENCES
---------
[1] Seshadri, S. & Räsänen, O. (2019). SylNet: An Adaptable End-to-End Syllable Count Estimator for Speech. IEEE Signal Processing Letters, 26, 1359–1363.
