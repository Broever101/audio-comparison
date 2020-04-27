# Doom-Eternal-Testosterone

First step: Download and install Anaconda Navigator to make life easier. 
https://docs.anaconda.com/anaconda/install/windows/

Next, open Anaconda and go to the Environments tab. In the base (root) environment, open terminal.
![Base Terminal](/Screenshot_320.png)

Type the following commands:
conda install numba  
conda install cudatoolkit  
pip install numpy  
pip install librosa  
pip install tqdm  


Now all the libraries are set up. In the directory where you put the ```tests.py``` file, create two folders: ```"source" and "target". ```  
Copy the files in KAREN folder in "source", or alternatively just copy the Karen folder and rename it to "source". 
Copy the game soundtrack in "target". 

The script will compare every file in target with every file in source, and generate a Matches.txt file with the following format:

target track, source track, offset, length of source track  

This tells us which target track matched with which source track and at what offset (the offset will be a fuzzy measure because we're not checking 
every possible cut).