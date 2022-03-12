
# SAINT
This repository contains the official implementation of our paper [***SAINT**: **S**elf-**A**ttention Augmented **I**nception-Inside-Inception **N**e**t**work Improves Protein Secondary Structure Prediction.*](https://doi.org/10.1093/bioinformatics/btaa531) published in the journal [Bioinformatics](https://doi.org/10.1093/bioinformatics/btaa531). 

If you use any part of this repository, we shall be obliged if you site [our paper](https://doi.org/10.1093/bioinformatics/btaa531).

# Usage
## Tensorflow and Keras installation
1. Please install [Tensorflow version: 1.15](https://www.tensorflow.org/install/gpu#older_versions_of_tensorflow). (Other 1.x versions should work, but have not been tested. Tensorflow-gpu version is recommended for faster inference.)
2. Please install [Keras version: 2.2.5](https://pypi.org/project/Keras/2.2.5/).

## Download pretrained-model weights:
1. Please download all the pretrained model weights from [here](https://drive.google.com/open?id=1mjXUfz33asJHBorEeMU0kd1A-1WChRyR) (4 weight-files for 4 single-models).
2. Place these four weight-files in the folder "SAINT".

## Input Features
To store input-features, navigate to the folder SAINT. Then follow any of the following two options:
### Option-1
1. List all the protein-names in the file: list_test

2. Place the FASTA, PSSM, HHM, and Spotcon (contact map generated by [Spot-Contact](https://sparks-lab.org/server/spot-contact/)) files for these proteins into the folder: Test

### Option-2
1. If you have a list of proteins in a text file (each in a new line) and a folder containing the FASTA, PSSM, HHM, and Spotcon files of these proteins, you can simply set their path in SAINT/config.py file.

## Run inference to predict secondary structures
1. From command line cd to "SAINT" folder (where SAINT_ensemble.py is situated).

2. In order to run inference with our complete SAINT (ensemble), please run the following command:

  > python SAINT_ensemble.py
  
3. If you want to run inference only with the FASTA, PSSM, and HHM (Without Spotcon), please run the following command:

  > python SAINT_single_base_model.py
  
# Output format
1. There are two types of output files: 
	- **Predicted Sequences:** Contain the predicted sequences of the secondary structures of the residues. The file extension is *".ss8"*.
	- **Predicted Probabilities:** Contain the predicted probabilities of the secondary structures, for each of the residues. The file extension is *".ss8_probab"*.
2. The predicted outout sequences and the predicted probabilities will be saved in separated files in the "SAINT/outputs" folder. The second of the third columns of the following table show the pattern of file-names of the output files, and the first column shows the types of the models: 

	| Model-type | File-name Patterns (Predicted Sequences) | File-name Patterns (Predicted Probabilities) |
	| -------------- |:--------------------------------------:|:--------------------------------------:|
	|*Ensemble*| *<Protein_Name>*.SAINT_Ensemble.ss8    | *<Protein_Name>*.SAINT_Ensemble.ss8_probab   |
	|*No Contact-Map*| *<Protein_Name>*.SAINT_cwin0.ss8    | *<Protein_Name>*.SAINT_cwin0.ss8_probab   |
	|*Contact Window size 10*| *<Protein_Name>*.SAINT_cwin10.ss8   | *<Protein_Name>*.SAINT_cwin10.ss8_probab   |
	|*Contact Window size 20*| *<Protein_Name>*.SAINT_cwin20.ss8   | *<Protein_Name>*.SAINT_cwin20.ss8_probab   |
	|*Contact Window size 50*| *<Protein_Name>*.SAINT_cwin50.ss8  | *<Protein_Name>*.SAINT_cwin50.ss8_probab  |
 3. In the above table, *<Protein_Name>* indicates the protein names given in file named *"list_test"*. Output files will be generated for all the proteins listed in this *"list_test"* file. 
 4. The characters in the output files of predicted sequences represent the eight states (Q8) as shown in the following table: 
 
 	|Character | State|
	| -------- |:----------:|
 	|H| α-helix |
	|G| 3_{10}-helix|
	|I| π-helix|
	|E| β-strand|
	|B| isolated β-bridge|
	|T| turn|
	|S| bend |
	|C| Others |
	
# Citation
Mostofa Rafid Uddin, Sazan Mahbub, M Saifur Rahman, Md Shamsuzzoha Bayzid, SAINT: self-attention augmented inception-inside-inception network improves protein secondary structure prediction, Bioinformatics, Volume 36, Issue 17, 1 September 2020, Pages 4599–4608, https://doi.org/10.1093/bioinformatics/btaa531

## BibTeX
```
@article{10.1093/bioinformatics/btaa531,
    author = {Uddin, Mostofa Rafid and Mahbub, Sazan and Rahman, M Saifur and Bayzid, Md Shamsuzzoha},
    title = "{SAINT: self-attention augmented inception-inside-inception network improves protein secondary structure prediction}",
    journal = {Bioinformatics},
    volume = {36},
    number = {17},
    pages = {4599-4608},
    year = {2020},
    month = {05},
    abstract = "{Protein structures provide basic insight into how they can interact with other proteins, their functions and biological roles in an organism. Experimental methods (e.g. X-ray crystallography and nuclear magnetic resonance spectroscopy) for predicting the secondary structure (SS) of proteins are very expensive and time consuming. Therefore, developing efficient computational approaches for predicting the SS of protein is of utmost importance. Advances in developing highly accurate SS prediction methods have mostly been focused on 3-class (Q3) structure prediction. However, 8-class (Q8) resolution of SS contains more useful information and is much more challenging than the Q3 prediction.We present SAINT, a highly accurate method for Q8 structure prediction, which incorporates self-attention mechanism (a concept from natural language processing) with the Deep Inception-Inside-Inception network in order to effectively capture both the short- and long-range interactions among the amino acid residues. SAINT offers a more interpretable framework than the typical black-box deep neural network methods. Through an extensive evaluation study, we report the performance of SAINT in comparison with the existing best methods on a collection of benchmark datasets, namely, TEST2016, TEST2018, CASP12 and CASP13. Our results suggest that self-attention mechanism improves the prediction accuracy and outperforms the existing best alternate methods. SAINT is the first of its kind and offers the best known Q8 accuracy. Thus, we believe SAINT represents a major step toward the accurate and reliable prediction of SSs of proteins.SAINT is freely available as an open-source project at https://github.com/SAINTProtein/SAINT.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa531},
    url = {https://doi.org/10.1093/bioinformatics/btaa531},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/36/17/4599/34220682/btaa531.pdf},
}
```
