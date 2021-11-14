# Earthquake-Waves-Classifier
Project for NTU CZ4042 Neural Network and Deep Learning

by Wilson Teng, Irvin Ting, Teo Kai Jie

## Dataset:   

Files to be used:
- 

Data:  
Labels (4 classes): 
- 

## EDA:
- Out of 656 audio files, check how many missing labels.
- Check how imbalanced our dataset is amongst the 4 classes.

## Preprocessing issues:
- Solve noisy data issue 
  - apply bandpass filter
- Solve varying amplitude btw diff audio files
  - normalisation
- Solve varying length issue, maybe we set at 5s
  - (> 5s) - randomly get 5s audio slice
  - (< 5s) - repeat audio til 5s reached
- Convert to spectrogram

## Potential roadblocks:
- Highly imbalanced dataset
  - For data: concat normal with abnormal audio
  - For eval: use stratified sampling
- Very noisy dataset
- Small dataset (many missing labels)
