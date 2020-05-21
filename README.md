# THIS PROJECT WAS ... DOOMED

The real motivation for this project was to find the right cuts in the Doom Eternal in-game soundtrack and replace them with Mick Gordon's own mixes, but the repacking of pck file turned out to be too ambitious (you can't unscramble an egg). So what we are left with is just a script that compares audio in `source` directory with audio in `target` directory, probabilistically and on GPU to save time.  

# DEPENDENCIES 

```
conda install numba  
conda install cudatoolkit  
pip install numpy  
pip install librosa  
pip install tqdm  
```
