# VowSynth

readFiles-main: This file is to preprocess input data, convert video to landmark points

train* : These files reads in input data, trains the network and save the generated test files.

audio_features: These files are used to compute WER on the train/test data. The file train an ASR on the groundtruth data and then evaluates the generated tokens.

run_metrics: These files are used to compute STOI/PESQ and MCD scores on the generated data.

![Synthesis Network](images/network-4.png)


[![Generated Videos](https://img.youtube.com/vi/th-eFkLCIQM/maxresdefault.jpg)](https://youtu.be/th-eFkLCIQM)
