# Cross Diffusion 
Changed source code for Interacting Diffusion Processes for Event Sequence Forecasting for my summer research internship. The original version was taken from https://github.com/networkslab/cdiff.

# Run the Code

## Utils
Copy the util files from the utils/ folder to the root directory and name it utils.py.
The different util files are:
- evalutils.py: Just for running the eval part by putting your own checkpoints and args
- null_all_tgt_utils.py: For running the eval part with null context
- train_for_null_utils.py: To train for null context evaluation
- utilsreal.py: The original utils file 
- 20_40_utils.py: To generate events 20 to 40 from the initial 0-20 events generated (In Progress)

## Scripts
The scripts are located in the scripts/ folder and must be copied to the root directory and then run.

## Jupyter Notebooks
- amazoneda.ipynb: Exploratory data analysis for the datasets
- amazon_convert_and_eda.ipynb: Code to convert the data formats from the one for ifl tpp to cdiff compatible form
- visual.ipynb: To visualise the generated sequence and compare it with original sequence
- visual.py: Same as above but in python script 

## Null Samples:

The folders contained in nullsamples/ are the generated sequences after taking as large a target length as possible.

- amazon20/: Contains 20 generated events from amazon 
- amazon88/ : Contains 88 generated events from amazon 
- retweet72/: Contains 72 generated events from retweet 
- so98/: Contains 98 generated events from stackoverflow
- taobao62/: Contains 62 generated events from taobao 
- taxi36/: Contains 36 generated events from taxi 

## Results:

The results for the above null context generations are as follows:

Amazon results: 
![Amazon results](results/amazon88.png)
We take a generated sequence length of 88 where 94 is the maxiumum sequence length in this dataset.
Retweet results:
![Retweet results](results/retweet72.png)
We take a generated sequence length of 72 where 97 is the maxiumum sequence length in this dataset.
Stackoverflow results:
![stackoverflow results](results/stackoverflow98.png)
We take a generated sequence length of 98 where 101 is the maxiumum sequence length in this dataset.
Taobao results:
![Taobao results](results/taobao62.png)
We take a generated sequence length of 62 where 63 is the maxiumum sequence length in this dataset.
Taxi results:
To be done.
We take a generated sequence length of 36 where 38 is the maxiumum sequence length in this dataset.


### Dependencies
```

SciPy  

Numpy   

scikit-learn 

seaborn

pytorch >= 1.8.0

PrettyTable

matplotlib
```

### Instructions
1. Put the data folder inside the root folder, modify the data entry in run.sh accordingly. The datasets are available 
2. We have provided you Taobao and Amazon dataset under folder ./data/taobao and ./data/amazon, for more datasets please go to [datasets](https://drive.google.com/drive/folders/1gT3fL5vJpLYPNtn9eGAbC7qrhmFgiyV0?usp=sharing)


# Credits
The following repositories are used in our code, either in close to original form or as an inspiration:
- [Multinomial Diffusion](https://github.com/ehoogeboom/multinomial_diffusion)
- [Attentive Neural Hawkes Process](https://github.com/yangalan123/anhp-andtt)
- [Intensity Free Temporal Point Process](https://github.com/shchur/ifl-tpp)




