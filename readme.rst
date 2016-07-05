``SparCC`` is a python module for computing correlations in compositional data (16S, metagenomics, etc').

Detailed information about the algorithm can be found in the accompanying `publication <http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002687>`__.  

Questions, comments, complaints and praise should be send to yonatanf@mit.edu



********************************
Usage Notes:
********************************
- Scripts in the root SparCC directory can be called from the terminal command-line either by explicitly calling python (as is done in the usage examples below), or simply as an executable. The latter will require having execution permission for these file (e.g. chmod +x SparCC.py).

- Help for any one for the scripts in the root SparCC directory is available by typing 'python [script_name] - h' in the command line. e.g.: :: 

   python SparCC.py -h .

- SparCC is implemented in pure python and requires a working version of python (=>2.3, tested with 2.6.6) and numpy (tested with version 1.10.1).

       
********************************
Usage example:
********************************
- The following lists the commands required for analyzing the included 'fake' dataset using the SparCC package, and generating all the files present in the subfolders of the example folder.

- The fake dataset contains simulated abundances of 50 otus in 200 samples, drawn at random from a multinomial log-normal distribution. The true basis correlations used to generate the data are listed in 'true_basis_cor.txt' in the example folder.

- Note that otu 0 is very dominant, and thus, using Pearson or Spearman correlations, appears to be negatively correlated with most other OTUs, though it is in fact not negatively correlated with any OTU.

---------------------------------
Correlation Calculation:
---------------------------------
First, we'll quantify the correlation between all OTUs, using SparCC, Pearson, and Spearman correlations:

::

   python SparCC.py example/fake_data.txt -i 5 --cor_file=example/basis_corr/cor_sparcc.out
   python SparCC.py example/fake_data.txt -i 5 --cor_file=example/basis_corr/cor_pearson.out -a pearson
   python SparCC.py example/fake_data.txt -i 5 --cor_file=example/basis_corr/cor_spearman.out -a spearman


---------------------------------
Pseudo p-value Calculation:
---------------------------------
Calculating pseudo p-values is done via a bootstrap procedure.
First make shuffled (w. replacement) datasets:
::

   python MakeBootstraps.py example/fake_data.txt -n 5 -t permutation_#.txt -p example/pvals/

This will generate 5 shuffled datasets, which is clearly not enough to get meaningful p-values, and is used here for convenience.
A more appropriate number of shuffles should be at least a 100, which is the default value. 

Next, you'll have to run SparCC on each of the shuffled data sets. 
Make sure to use the exact same parameters which you used when running SparCC on the real data, name all the output files consistently, numbered sequentially, and with a '.txt' extension.
::

   python SparCC.py example/pvals/permutation_0.txt -i 5 --cor_file=example/pvals/perm_cor_0.txt
   python SparCC.py example/pvals/permutation_1.txt -i 5 --cor_file=example/pvals/perm_cor_1.txt
   python SparCC.py example/pvals/permutation_2.txt -i 5 --cor_file=example/pvals/perm_cor_2.txt
   python SparCC.py example/pvals/permutation_3.txt -i 5 --cor_file=example/pvals/perm_cor_3.txt
   python SparCC.py example/pvals/permutation_4.txt -i 5 --cor_file=example/pvals/perm_cor_4.txt

Above I'm simply called SparCC 5 separate times. However, it is much more efficient and convenient to write a small script that automates this, and submits these runs as separate jobs to a cluster (if one is available to you. Otherwise, this may take a while to run on a local machine...).

Now that we have all the correlations computed from the shuffled datasets, we're ready to get the pseudo p-values.
Remember to make sure all the correlation files are in the same folder, are numbered sequentially, and have a '.txt' extension.
The following will compute both one and two sided p-values.
::

   python PseudoPvals.py example/basis_corr/cor_sparcc.out example/pvals/perm_cor_#.txt 5 -o example/pvals/pvals.one_sided.txt -t one_sided
   python PseudoPvals.py example/basis_corr/cor_sparcc.out example/pvals/perm_cor_#.txt 5 -o example/pvals/pvals.one_sided.txt -t two_sided


