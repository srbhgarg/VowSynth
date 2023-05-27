# VowSynth

## Abstract ##
Humans use both auditory and facial cues to perceive speech, especially when auditory input is degraded, indicating a direct association between visual articulatory and acoustic speech information. This study investigates how well an audio signal of a word can be synthesized based on visual speech cues. Specifically, we synthesized audio waveforms of the vowels in monosyllabic  English words from motion trajectories extracted from image sequences in the video recordings of the same words. The articulatory movements were recorded in two different speech styles: plain and clear. We designed a deep network  trained on mouth landmark motion trajectories on a spectrogram and formant-based custom loss for different speech styles separately. Human and automatic evaluation show that our framework using visual cues can generate identifiable audio of the target vowels from distinct mouth landmark movements. Our results show that the intelligible audio can be synthesized on unseen speakers that were not part of the training data.

## Notes ##

readFiles-main: This file is to preprocess input data, convert video to landmark points

train* : These files reads in input data, trains the network and save the generated test files.

audio_features: These files are used to compute WER on the train/test data. The file train an ASR on the groundtruth data and then evaluates the generated tokens.

run_metrics: These files are used to compute STOI/PESQ and MCD scores on the generated data.

![Synthesis Network](images/network-4.png)


[![Generated Videos](https://img.youtube.com/vi/th-eFkLCIQM/maxresdefault.jpg)](https://youtu.be/th-eFkLCIQM)


lip2Audspec: data preparation, training autoencoder and training lip2aud network. lip2audspec_autoencoder.ipynb

pipeline.sh: the complete pipeline
1. python generate_style.py $input_file  "demographics"
2. nohup python face_detector_mtcnn.py <filename_list>  &
3. align=True/False; nohup python eval_san_final.py Deep-structured-facial-landmark-detection/coord_file.txt  &
4. or cd Deep-structured-facial-landmark-detection/; source tensorflow1/bin/activate; sbatch --array=1-30 job.sh 
5. Set opt in create_data.py; python create_data.py
6. nohup python train_audiosynthesis.py
7. python -i test_audiosynthesis.py
8. python -i get_asr.py

lpc and compute_formants: contains implementation of formants in tensorflow.

Note: The videos in the "videos" folder are sample generated audios by the model from the lip movements that are then overlaid on the actual video manually.

## Reference ##
if you use this for research publications, please cite:
```
@article{article,
year = {2023},
month = {05},
pages = {1-16},
title = {Mouth2Audio: intelligible audio synthesis from videos with distinctive vowel articulation},
journal = {International Journal of Speech Technology},
doi = {10.1007/s10772-023-10030-3}
}
```
https://link.springer.com/article/10.1007/s10772-023-10030-3
