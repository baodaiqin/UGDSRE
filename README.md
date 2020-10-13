# UGDSRE
Codes and datasets for our paper "Two Training Strategies for Improving Relation Extraction over Universal Graph"
## Dependencies
- python = 2.x
- tensorflow = 1.9.0
- numpy = 1.15.4
- sklearn = 0.19.1
## Data preprocessing
### NYT10 dataset
- We use NYT10 (Riedel et al., 2010) and Biomedical datasets for evaluation.
- The datasets with path evidences can be downloaded from here: NYT10 and Biomedical.
Unzip the NYT10 dataset and allocate the files under the directory, then run
~~~
python initialize_nyt10_part1.py
python initialize_nyt10_part2.py
~~~
### Biomedical dataset
Unzip the Biomedical dataset and allocate the files under the directory, then run
~~~
python initialize_biomedical_part1.py
python initialize_biomedical_part2.py
~~~
## Evaluate pretrained model
## The results of the pretrained model
## Train our model
