# Computer Vision - AV HuBERT Research

## About

This is my submission for the Computer Vision module for the University
of Portsmouth MEng Computer Science course. This research explores using
the SOTA AV-HuBERT model, which is used for ASR for Lip Reading and also
works as a useful feature extractor for input video frames audio-visual tasks.

This research explores using these extracted features for phoneme prediction
and mel-spectrogram synthesis. The phoneme classification confusion is assessed
to determine where classifiers fall short.

### Directory Breakdown

- `av_hubert/`: Meta AV HuBERT submodule
- `split.py`: Splits a source MP4 video into 10 second clips. This is because
  the AV HuBERT model works best with up to 10 second clips.
- `main.ipynb`: Contains all of the initial experimental code for this project...
  - AV HuBERT Feature Extraction (Base, Self-Trained Large): Generate features
    for 10 second clips
  - SKLearn and PyTorch classifier training code
  - Dataset Handling Code (Load phonemes, audio features, raw dlib facial
    landmarks, OpenAI Whisper Large word-level timestamps)
  - Auxilliary mel spectrogram prediction experiments for more robust training
- `base_vox_433h`: AV HuBERT BASE model
- `self_large_vox_433h`: Self-Trained AV HuBERT LARGE model (Best performing)
- `phoneme_dict.txt`: ARPABET phoneme dictionary

<!-- ### GPT-4 Phoneme "Beam Search"

The work also explores using GPT4 as a form of phoneme beam search. This
work hypothesizes that GPT4's ability to compose phonemes together into
sensible sentences based on GPT4's large training dataset and ability
to contextually understand phonemes and generate plausable sentences
gives it an advantage over existing methods. -->

### Classifiers

Different classifiers are explored over the AV HuBERT features:
1. PyTorch Deep Neural Network (1 hidden layer deep neural network with 256 or 512 hidden dimension
   and ReLU activation and then softmax)
2. Support Vector Machine (Linear, Radial Basis, Poly, Sigmoid)
3. Random Forest

### Visual Features

This work explores two main types of visual features:
1. `AV HuBERT` Embeddings (Generated from their VoxCeleb3, 
   fine tuned model `base_vox_433h`, `self_large_vox_433h`.)
   - BASE  (768 dim)
   - Self-Trained LARGE (1024 dim)
2. Base `dlib` facial landmarks

### Datasets

Two datasets are used for this work:
<!--
1. VoxCeleb (This is a standard dataset used in Lip Reading research as it
   contains many videos with the lips of celebrities speaking clearly visible.
   LRS3 is also considered, but not explored in this work as written consent
   needs to be obtained before using this dataset for research.).
   As the AV HuBERT model used in this work has been fine tuned on this dataset,
   only the test set portion of this dataset is used for evaluation for fairness.
   VoxCeleb2 was considered at first, but the audio visual dataset is restricted
   and requires a password.
-->
1. Jordan Peterson Lecture (30fps)
   This dataset has a duration of ~= 11 mins and 24 secs or 684 secs and a sequence length of
   ~20,000 frames.
2. Jordan Peterson (24fps) (The False Appeal of Communism) (Shorts clip of Jordan Peterson discussing
   communism. Good clip to use due to variety of phonemes present within the
   dataset.)
   This dataset has a duration of ~= 51 seconds and a sequence length of 1,233 frames.
<!--
3. Personal Dataset (This is a personal dataset used for initial experiments
   with a mixture of celebrities speaking, with the videos being chosen for
   the varieties of phonemes expressed during the videos.)
-->