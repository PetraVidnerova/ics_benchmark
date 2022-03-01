# ics_benchmark


# Setup 

Create conda environment adn install required libraries.

```sh
conda create -n benchmark python=3.9
conda activate benchmark
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
conda install click tqdm
```

# Data
TODO FIX ME
```
wget -c https://filr.cs.cas.cz:443/ssf/s/readFile/share/226/-7042450145272094661/publicLink/data.tgz
md5sum data.tgz
```
6b4d948b0402caa2b61af5270138b357  data.tgz
    
```
tar xvfz data.tgz 
```

# Run

One GPU variant: 
```
python benchmark.py --data_root <PATH_TO_DATA> 
```

Two GPUs variant:
```
python dist_benchmark.py --data_root <PATH_TO_DATA>
```
