# IMAGE Conductance Analyzer

`icAnalyzer` is a tool for carrying out statistical analysis on ionospheric conductance estimates based on the IMAGE satellite mission and made using [`icBuilder`](https://github.com/BingMM/icBuilder).

Estimated ionospheric conductances with associated uncertainties are available [**here**](https://doi.org/10.5281/zenodo.15579301).

## Project Description

The main purpose of this codebase is to document the data processing procedure. While not primarily designed for external use, the code can be run by others if needed.

## Dependencies

- [`icReader`](https://github.com/BingMM/icReader) - for reading conductance output files

## Installation

mamba activate your_environment  
git clone https://github.com/BingMM/icReader.git  
cd icReader  
pip install -e .

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or comments, please contact [michael.madelaire@uib.no].
