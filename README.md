
This repository contains code for the research paper

**Necessary and Sufficient Conditions for Existence and Uniqueness of Recursive Utilities**

by Jaroslav Borovička and John Stachurski.

The paper is available [via arXiv](https://arxiv.org/abs/1710.06526).

At this stage the code is written only in Python and Julia.  MATLAB code is in
the pipeline.  (Please prompt us if you need it.)

The Python code is slower but it's also clearer and more stable.  (The only
issue with speed is that we need to recompute the stability coefficients at
many different parameterizations.  This operation is highly
parallelizable but any such optimizations make the code less transparent and
less portable.  Hence we've stuck with the base implementation.)

The figures are reconstructed in Jupyter notebooks.  If you navigate to the
`*.ipynb` files in the directories above you will be able to see them.

The whole scientific Python stack can be downloaded as a package
[here](https://www.anaconda.com/download/).  With that package installed the
Python code should run via the Jupyter notebook application.

Please feel free to contact the authors if you have questions.
