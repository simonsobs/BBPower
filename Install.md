# Installation of PolyChord

##### At NERSC
Load the modules
```
module load python/3.8-anaconda-2020.11
module load PrgEnv-intel
```
Then, pypolychord successfully installs with
```bash
pip install pypolychord
```

This is also compatible with using a conda environment with python 3.8. 


##### On Linux laptop (Ubuntu 18.04)

`pip install pypolychord` requires a conda environment with python 3.6 to work successfully. If this does not work, try

```
git clone https://github.com/PolyChord/PolyChordLite.git
cd PolyChordLite
python setup.py install
```

See also https://github.com/PolyChord/PolyChordLite.

