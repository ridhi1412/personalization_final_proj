### Analysis of diffrent recommendation engines for business recommendation in Yelp
#### Personalization: Theory & Applications Final Project
#### IEOR 4571
#### Fall 2019
_____________________________________________________________________________________________________________________________


##### Authors: 
1. [Ridhi Mahajan](https://github.com/rmahajan14) (rm3601)
2. [Sheetal Reddy](https://github.com/Sheetalkreddy) (kr2793)
3. [Chandrasekaran Anirudh Bhardwaj](https://github.com/anirudhsekar96) (cb3441)

_____________________________________________________________________________________________________________________________

> In this project we perform an in-depth analysis of different algorithms for business recommendation in Yelp data. 

The report is in [Report.pdf](./Report.pdf)

Code is structured as follows

    .
    ├── data                            # Load data & Sampling functions
    |     ├── results
    |     |      └── csv files for results			
    │     └── data files			
    |
    ├── python
    |     ├── common.py                 # Data Loading and utility functions
    |     ├── bias.py			        # Bias based model
    |     ├── colab_filtering.py		# Colaborative Filtering based model - ALS
    |     ├── content_based.py			# Content Based Reccomendation - based on Tf-IDF 
    │     ├── Deep_learning.py		    # Deep Learning architectures
    │     ├── hermes.py                 # https://github.com/Lab41/hermes
    │     ├── Data Load.ipynb           # Data Engineering code for combining user and business meta-data to ratings
    │     ├── Location and Time aware data.ipynb # Data Engineering for location and time based discounting of ratings
    │     ├── EDA.ipynb                 # Exploratory Analysis of Data and Results
    │     ├── 
    |     └── 
    ├── graphs                     			# Graph files and base files for wordcloud
    ├── cache                    			# Data cache used to avoid re-reading the data each time model changes are made
    ├── requirments.txt                     # pip requirment files to reproduce the results
    ├── Report.pdf                   		# Report File
    ├── LICENSE
    └── README.md



