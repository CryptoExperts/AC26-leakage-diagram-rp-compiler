
[comp_bounds.py]:comp_bounds.py
[results.tex]:results.tex


# AC26 - Leakage Diagram RP Compiler

This repository contains all the scripts necessary to reproduce Figure 7 
presented in the paper:

"Refining Leakage Diagram Analyses for Random Probing Security" by Sonia Belaïd,
Ghozlane Boukacem, Gaëtan Cassiers, Victor Normand, and Mélissa Rossi, published at Asiacrypt 2026.

In the following, we may refer to the full version of this publication, which includes an appendix to the main text. The full version is available here:

**TODO**


## Dependencies

This project requires : 
  - Python 3.x
  - `numpy`, `matplotlib`, `mpmath`
  - A LaTeX distribution providing `pdflatex`, `TikZ`, and `pgfplots`

To install dependencies, run :

`pip install numpy matplotlib mpmath`

## Usage

To obtain the Figure 7 of the full version of the paper, please run the
following command : 

`python3 comp_bounds.py`

`pdflatex results.tex`


If you intend to change any parameter values, check the function’s documentation 
in [comp_bounds.py] for a better understanding of what each parameter does.


## Organization of the Repository

  1. [comp_bounds.py] : This file enables the computation of the Random Probing
     security advantage $\varepsilon$ for some leakage rate $p$ and number of
     shares $n$. This is done for different circuit compilers. Namely  : 
     - The [BFO23] compiler, which is the original work based on Leakage
       Diagrams, the corresponding paper is available at : https://eprint.iacr.org/2023/1182
     - The [JMB24] compiler, the corresponding paper is available at :
       https://tches.iacr.org/index.php/TCHES/article/view/11806
     - The [BNR25] compiler, the corresponding paper is available at :
       https://eprint.iacr.org/2025/1747
      
     The random probing security results are then stored in tsv file. This file
     also enables the complexity computation of these compilers.

  2. [results.tex] : This file loads the different tsv files computed using
     [comp_bounds.py] as well as some others computed using PERSEUS tool
     (i.e. [BC26], see https://eprint.iacr.org/2025/1884 for more information)
     and produces a pdf which is the Figure 7 of the paper. 

  

## License

This project is distributed under the **MIT License**.

##

For any questions, please refer to the paper or contact the authors.
 


  
  




  
  
  
  
