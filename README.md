## Introduction
* This repository provides implementations of Ex-DPC++.
* This is a fast algorithm for [density-peaks clustering](https://science.sciencemag.org/content/344/6191/1492.full) (proposed in Science).
* As for the detail about this algorithm, please read our TKDD paper, [Fast Density-Peaks Clustering for Static and Dynamic Data in Euclidean Spaces](https://dl.acm.org/doi/).

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
@article{amagata202xdpc,  
    title={Fast Density-Peaks Clustering for Static and Dynamic Data in Euclidean Spaces},  
    author={Amagata, Daichi and Hara, Takahiro},  
    booktitle={ACM Transactions on Knowledge Discovery from Data},  
    pages={xx--xx},  
    year={202x}  
}
```

## License
Copyright (c) 2023 Daichi Amagata  
This software is released under the [MIT license](https://github.com/amgt-d1/Ex-DPC/blob/main/license.txt).
