## Introduction
* This repository provides implementations of Ex-DPC++, an extended version of [Ex-DPC](https://github.com/amgt-d1/DPC).
* This is a fast algorithm for [density-peaks clustering](https://science.sciencemag.org/content/344/6191/1492.full) (proposed in Science).
* As for the detail about this algorithm, please read our TKDD paper, [Efficient Density-Peaks Clustering Algorithms on Static and Dynamic Data in Euclidean Space](https://dl.acm.org/doi/10.1145/3607873).
	* Different from the setting in this paper (orthogonal range), the implementation in this repository assumes circular range because the original DPC paper assumes this setting. In addition, we prepared a VP-tree to deal with arbitrary metric spaces.

## Requirement
*  [Eigen](https://eigen.tuxfamily.org/)
* The source code of the DPC algorithm may need to be changed based on your paths of the above library.

## How to use
* We assume low-dimensional datasets, as we use a tree structure.

### Linux (Ubuntu)
* Compile: `g++ -O3 main.cpp -o exdpc.out -fopenmp` and run: `./exdpc.out`.

### Datasets
* As an example, we have prepared a 2-dimensional synthetic dataset used in our paper.
* If you want to test your dataset,
	* Put the file at `dataset` directory.
	* Assign a unique dataset ID.
	* Set the dimensionality at `input_parameter()` of `exdpc.hpp`.  
	* Update `input_data()` function of `exdpc.hpp`.  
	* Add a directory in `result` and update the function `output_result()` of `exdpc.hpp`.
	* Compile the code and run .out file.

### Parameters
* Set some value in the corresponding txt file in `parameter`.
* For \rho_min and \delta_min, we specify them in `input_parameter()` of `exdpc.hpp`.

## Citation
If you use our implementation, please cite the following paper.
``` 
@article{amagata2024dpc,  
title={Efficient Density-Peaks Clustering Algorithms on Static and Dynamic Data in Euclidean Spaces},  
	author={Amagata, Daichi and Hara, Takahiro},
	booktitle={ACM Transactions on Knowledge Discovery from Data},
	volume={18},
	number={1},
	pages={2:1--2:27},
	year={2024}
}
```

## License
Copyright (c) 2023 Daichi Amagata  
This software is released under the [MIT license](https://github.com/amgt-d1/Ex-DPC-plus-plus/blob/main/LICENSE).
