# IR2_5
Project for Information Retrieval 2, team 5
## Contributors
- Tom Lotze
- Stefan Klut

## Project discription:
A new development in online search engine search is clarification questions, posed to to the user to clarify the information need. Using the MIMICS dataset we aim to predict the user engagement from lexical information about the query, question and answers using a neural regression model. Afterwards, we investigate whether these predicted user engagements can help in learning to rank the various possible clarification questions. 

## Dataset:
To run the code the MIMICS dataset should be present in a folder "./Data/". The MIMICS dataset can be downloaded from https://github.com/microsoft/MIMICS.

## Usage:
To create a preprocessed version of the data run the dataloader.py:

    python code/dataloader.py -h
    
To train any of the models, run train_{modelname}.py:

    python code/train_{modelname}.py -h

Scripts for running the preprocessing and training on Surflisa is available in the "scripts/" folder.
