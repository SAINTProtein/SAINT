
# SAINT
This repository contains the official implementation of our paper **SAINT**: **S**elf-**A**ttention Augmented **I**nception-Inside-Inception **N**e**t**work Improves Protein Secondary Structure Prediction.

If you use any part of this repository, we shall be obliged if you site our paper [**SAINT**: **S**elf-**A**ttention Augmented **I**nception-Inside-Inception **N**e**t**work Improves Protein Secondary Structure Prediction](https://www.biorxiv.org/content/10.1101/786921v1).

# Usage
To store input features, navigate to the folder SAINT. Then follow any of the following two options:
## Option-1
1. List all the protein-names in the file: list_test

2. Place the FASTA, PSSM, HHM, and Spotcon (Contactmaps, output of the Spot-Contact) files for these proteins into the folder: Test

## Option-2
1. If you have a list of proteins in a text file (each in a new line. Last line of the file is a blank line) and a folder containing the FASTA, PSSM, HHM, and Spotcon files of these proteins, you can simply set their path in SAINT/config.py file.

## Run inference
1. From command line cd to SAINT folder (where SAINT_ensemble.py is situated).

2. Then run the following command:

  > python SAINT_ensemble.py
  
# Output format
1. The predicted outout sequences will be saved in the following files in the "SAINT/outputs" folder: 
	-Ensemble: SAINT_Ensemble_output_ss8_sequences.txt  
	-Contact Window size 0: SAINT_cwin0_output_ss8_sequences.txt  
	-Contact Window size 10: SAINT_cwin10_output_ss8_sequences.txt  
	-Contact Window size 20: SAINT_cwin20_output_ss8_sequences.txt  
	-Contact Window size 50: SAINT_cwin50_output_ss8_sequences.txt  
 2. Each new line contains Protein 8 state secondary structure for each protein in the above-mentioned list (in the same order).
