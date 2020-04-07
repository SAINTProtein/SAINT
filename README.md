
# SAINT
This repository contains the official implementation of our paper *"**SAINT**: **S**elf-**A**ttention Augmented **I**nception-Inside-Inception **N**e**t**work Improves Protein Secondary Structure Prediction."*

If you use any part of this repository, we shall be obliged if you site our paper [**SAINT**: **S**elf-**A**ttention Augmented **I**nception-Inside-Inception **N**e**t**work Improves Protein Secondary Structure Prediction](https://www.biorxiv.org/content/10.1101/786921v1).

# Usage

## Download pretrained-model weights:
1. Please download all the pretrained model weights from [here](https://drive.google.com/open?id=1mjXUfz33asJHBorEeMU0kd1A-1WChRyR) (4 weight-files for 4 single-models).
2. Place these four weight-files in the folder "SAINT".

## Input Features
To store input-features, navigate to the folder SAINT. Then follow any of the following two options:
### Option-1
1. List all the protein-names in the file: list_test

2. Place the FASTA, PSSM, HHM, and Spotcon (contact map generated by [Spot-Contact](https://sparks-lab.org/server/spot-contact/)) files for these proteins into the folder: Test

### Option-2
1. If you have a list of proteins in a text file (each in a new line. Last line of the file is a blank line) and a folder containing the FASTA, PSSM, HHM, and Spotcon files of these proteins, you can simply set their path in SAINT/config.py file.

## Run inference to predict secondary structures
1. From command line cd to "SAINT" folder (where SAINT_ensemble.py is situated).

2. In order to run inference with our complete SAINT (ensemble), please run the following command:

  > python SAINT_ensemble.py
  
3. If you want to run inference only with the FASTA, PSSM, and HHM (Without SpotCon), please run the following command:

  > python SAINT_single_base_model.py
  
# Output format
1. The predicted outout sequences will be saved in the following files in the "SAINT/outputs" folder:    
	-*Ensemble*: *<Protein_Name>*.SAINT_Ensemble.ss8     
	-*No Contact-Map*: *<Protein_Name>*.SAINT_cwin0.ss8    
	-*Contact Window size 10*: *<Protein_Name>*.SAINT_cwin10.ss8   
	-*Contact Window size 20*: *<Protein_Name>*.SAINT_cwin20.ss8   
	-*Contact Window size 50*: *<Protein_Name>*.SAINT_cwin50.ss8  
 2. Hence *<Protein_Name>* indicates the protein names given in the *list_test* file. Similar Outputs will be generated for all the proteins mentioned in the list.  
