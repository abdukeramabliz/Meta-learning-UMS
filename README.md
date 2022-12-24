#  Meta-learning method of Uyghur morphological segmentation
## Contents
* [Introduction](#introduction)
* [Usage](#usage)
* [Experimental Results](#experimental-results)
* [License](#License)
* [Citation](#Citation)
* [Development Team](#development-team)
* [Contributors](#contributors)
* [Contact](#contact)

## Introduction

Morphological segmentation is a natural language processing task that aims to segment to its' corresponding morphemes, which are smallest meaning unit. 
This is the meta-learning method of Uyghur morphological segmentation

## Usage
In the trunk folder

Preprocessing

```
python load_data.py
```

Training

```
python train.py
```



## Experimental Results

5-way 1-shot 
| Method | P | R | F |
| :------------: | :---: | :--------------: | :----------------: |
| BERT       |  79.44 | 79.44 | 79.48 | 
| mBERT       |  80.21 | 80.21 | 80.50 | 
| InfoXLM       |  70.96 | 70.96 | 68.05 | 
| Ours      |  81.40 | 81.40 | 80.50 | 


5-way 5-shot
| Method | P | R | F |
| :------------: | :---: | :--------------: | :----------------: |
| BERT       |  84.15 | 84.15 | 83.69 | 
| mBERT       |  85.52 | 85.52 | 85.30 | 
| InfoXLM       |  77.72 | 77.72 | 76.12 | 
| Ours      |  85.65 | 84.40 | 84.78| 



## License

The source code is dual licensed. Open source licensing is under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause), which allows free use for research purposes. 

## Citation

Please cite the following paper:

> ZHANG Yuning,LI Wenzhuo,Khalidanmu Abdulkrim,Abdulkrim Abdulkiz. [Meta-learning method of Uyghur morphological segmentation](https://kns.cnki.net/kcms/detail/11.2127.tp.20220524.1512.020.html).Computer Engineering and Applications:1-8[2022-12-24]. (in Chinese)

## Development Team

Project leaders: Abudukelimu Halidanmu, Abulizi Abudukelimu

Project members: Zhang Yuning, Li Wenzhuo

## Contributors 
* [Abudukelimu Halidanmu](mailto:abdklmhldm@gmail.com) 

## Contact

If you have questions, suggestions and bug reports, please email [abdklmhldm@gmail.com](mailto:thumt17@gmail.com).
