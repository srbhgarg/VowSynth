# VowSynth

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

lpc and compute_formants: contains implementation of formants in tensorflow.
