# Deep Transfer Learning
**Electronic medical records (EMR)-based infectious disease case detection often faces performance drop in a new setting, which may result from differences in population distribution, EMR system implementation, and hospital administrations. Considering both the similarities and the dissimilarities between a source setting and a target setting, transfer learning may offer an effective way of improving the re-usability of source knowledge in a target setting. This study aims to explore when and how deep transfer learning is useful for infectious disease case detection.**  

**We simulated multiple transfer scenarios that vary in the target training size and the dissimilarity between the source and target settings (measured by the Kullback–Leibler divergence, KL). We compared Domain adversarial neural networks (DANN), a classic source data-based deep transfer learning method, source model-based deep transfer learning (MDTL), and baseline models, including a source model, a target model, and a combined model that was developed using the combination of source and target training data. We have summarized our research findings in a manuscript, which will be submitted to a peer-reviewed journal soon.**

**Through this GitHub repository, we publicly share the main research codes and simulated datasets so that other researchers can leverage them for further analysis. Our codes are derived from a transfer learning package by [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library), which is under MIT license.**



----

**In the following section, we use this README.md file to briefly explain how our code works. Further information can be found in the to-be-published manuscript and related references.**



### Configuration  
Language
- [x] Python (version--3.8)
 
Package:
- [x] PyTorch (version--1.7.1) 
- [x] torchvision (version--0.8.2)  
- [x] qpsolvers (version--1.8.0)

The configuration can be set by `conda` as below:  
```
conda create --name TL python=3.8 anaconda
conda activate TL
conda install pytorch==1.7.1 torchvision==0.8.2  -c pytorch
pip install qpsolvers==1.8.0
```

(Note: If you are using torchvision and pytorch that are not compatible with specific version of Python, or if your version of torchvision is not compatible with your version of PyTorch, you may face some tricky bugs when running our code.)

For example, if you are using a different version of pytorch or torchversion than we are, you may need to make some changes in the import statement section, for example.     
Change the
```
from torchvision.models.utils import load_state_dict_from_url
```
into  
```
from torch.hub import load_state_dict_from_url
```

To avoid unexpected bugs, we recommend that you run our code in the same configuration as we did.

### Download and Run

```
git clone https://github.com/AI-Public-Health/Deep-Transfer-Learning-Package.git

cd Deep-Transfer-Learning-Package/code/code
```
Run the all of models in the repositiry. For illustration, in here, the source data is specified as `findings_final_0814_seed1591536269_size1000.csv` (stored in */synthetic_data_v2/source_train*), target data is `findings_final_0814_seed-53154026_size50.csv`(stored in */synthetic_data_v2/target_train*), initial random seed is `14942`, number of epoch is `1`. You can change these hyperparameters to suit your needs.   
If you run `main_run.py` directly, without setting any parameters, the program will use all defaults (run on all datasets in */synthetic_data_v2*, trying several predefined random seeds), which may take some time to complete the whole process. 
```python
python main_run.py --source=findings_final_0814_seed1591536269_size10000 --target=findings_final_0814_seed-53154026_size50 --seed=14942 --epoch=1
```





# Table of contents
1. [Data processing](#dataprocess)
2. [Data-based Deep Transfer Learning](#DDTL)
    1. [DANN--unsupervised](#DDTL_unsupervised)
    2. [DANN--supervised](#DDTL_supervised)

3. [Model-based Deep Transfer Learning ](#MDTL)

4. [Baseline Model](#BL)

5. [Main Results and Conclusion](#conclusion)


> ## Data pre-processing  <a name="dataprocess"></a>
In the transfer learning scenarios, the source and target data can be recorded/presented differently. Since deep neural network requires the same variable dimension in the source and target setting, the data pre-processing is important.

*  **Kullback-Leibler divergence(KL)**      
In this project, when the source and target data have the same variable dimension, we use Kullback-Leibler (KL) divergence to measure the distribution difference between the source and target settings. Because the main experiments are conducted on synthetic datasets, we just release these datasets in our */data/synthetic_data_v2/* directory.      
The definition of KL-divergence is shown below (*t*  represents target, *s* represents source ).
![image](https://user-images.githubusercontent.com/39432361/152086546-b42438da-7f0e-411a-b08f-89cc445af061.png)
As description in our paper, the caculation of KL can be simplfied in this context.
![image](https://user-images.githubusercontent.com/39432361/152859024-3b78dc55-1927-4c68-bdb2-44d014140e3c.png)  
 In our code, we use `parse_model.py` to generate probability tables of difference datasets, use `distribution.py` to calculate the KL value between probability tables derived from different data sources.


* **Zero Padding**   
Data in the source setting may have/often has different dimensions than data extracted from the target setting. Therefore, before building models, the source and target data should be mapped into the same space. Our program solve this problem by **padding extra zeros** so that data from different settings have the same dimensionality.   
Even though all the data in the released synthetic datasets have the same dimensionality in both the source and target settings, we still include this **zero padding** method in `data_preprocess.py` for situations where you might have data with different dimensions.

After this recalibration of the data, we use all of the source data for model training and split our target data into different portions, for training, validation, or testing in the following models.   


* **Data Splitting**  
During experiments, the dataset needs to be splitted into different portions for difference purpose--training, testing, validation. In `data_procesing.py`, the associated methods are defined to split the dataset.  
In our experiments, due to the synthetic data being balanced distributed, and the dataset is split in the order of rows. However, if your dataset is skewed, you might need to change our methods of data splitting into stratified splitting ones, which are also included in `data_procesing.py`.        
(e.g.: changing `prepare_datasets_returnSourceVal` to `prepare_datasets_stratify_returnSourceVal`)

&nbsp;






> ## Data-based Deep Transfer Learning (DDTL) <a name="DDTL"></a>
**In DDTL section,  we apply the Domain Adversarial Neural Networks [(DANN)](https://arxiv.org/abs/1505.07818)**.  
DANN works in three scenarios: the target training data have class labels (DANN_supervised); the target training data do not have any class label (DANN_unsupervised); a portion of the target training data have class labels (DANN_semisupervised). We have two separated code file for the first two scenarios. For the third scenario, users need to adjust the code or the structure of input data accordingly. 
For Domain Adversarial Neural Networks(DANN), the *Loss function* is defined as below:
![image](https://user-images.githubusercontent.com/39432361/152067006-54cb0fef-d557-47f0-81eb-de2f3ddc39d4.png)


In our code, we use `dann_synthetic_noTargetLabel_noTargetVal.py`, `dann_synthetic_noTargetLabel_noTargetVal_outputAUC.py`, (the first two files are for unsupervised DANN),`dann_synthetic_withTargetLabel.py`, `dann_synthetic_withTargetLabel_outputAUC.py` (the last two files are for supervised DANN) to train and evaluate DDTL models. View our work in a whole picture, models are fed by source and target data files (.cvs) in ***synthetic_data_v2*** directory. After training, trained models will be stored. The program returns and saves a .txt file recording information of the source, target data, accuracy, AUC of models during training.

For the network structure, it is defined in `feedforward.py`. If you want to modify the network structure used in the DANN model, you can modify the corresponding part of that .py file.

&nbsp;    
> ### i. DANN(unsupervised) <a name="DDTL_unsupervised"></a>
Related code: 
`dann_synthetic_noTargetLabel_noTargetVal.py`, `dann_synthetic_noTargetLabel_noTargetVal_outputAUC.py`

In the unsupervised DANN, the Label Classifier is only trained by source training data (target training data is not to be used for the training of Label Classifier). Both target and source data are used to train the Feature Generator and the Domain Classifier.



Run the following code to train the DDTL model under an unsupervised setting---all of input data (.csv files of source and target data) are declared in the `if __name__ == '__main__'` module.
```python 
#run program using default hyperparameters---train all models with different combination of source and target datasets
python dann_synthetic_noTargetLabel_noTargetVal.py

#run program with specified learning rate (0.02), specified trade-off(3), specified source dataset(findings_final_0814_seed1591536269_size10000.csv), specified target dataset(indings_final_0814_seed-53154026_size50.csv)
python dann_synthetic_noTargetLabel_noTargetVal.py --lr=0.02 --trade-off=3 --source=findings_final_0814_seed1591536269_size10000 --target=findings_final_0814_seed-53154026_size50
``` 

Run the following code to return the corresponding `AUROC` for later performance in the comparison section---all of input data (.csv files of source and target data) are defideclaredned in the `if __name__ == '__main__'` module.
```python 
#run program using default hyperparameters---calculate AUC values from all of the models derived from different combinations of target and source dataset
python dann_synthetic_noTargetLabel_noTargetVal_outputAUC.py

#run program with specified learning rate (0.02), specified trade-off(3),specified source dataset(findings_final_0814_seed1591536269_size10000.csv), specified target dataset(indings_final_0814_seed-53154026_size50.csv).
python dann_synthetic_noTargetLabel_noTargetVal_outputAUC.py --lr=0.02 --trade-off=3 --source=findings_final_0814_seed1591536269_size10000 --target=findings_final_0814_seed-53154026_size50
``` 
&nbsp;

> ### ii. DANN (supervised)  <a name="DDTL_supervised"></a>
Related code: 
`dann_synthetic_withTargetLabel.py`, `dann_synthetic_withTargetLabel_outputAUC.py`

 In the supervised DANN, both source and target training data is be used in training the Label Classifier. Both target and source data are used in training Feature Generator and Domain Classifier. 

 In our project, we run the following codes to train the DDTL model under an unsupervised setting----all of input data (.csv files of source and target data) are declared in the `if __name__ == '__main__'` module.
```python 
#run program using default hyperparameters---train all models with different combination of source and target datasets
python dann_synthetic_withTargetLabel.py

#run program with specified learning rate (0.02), specified trade-off(3), specified source dataset(findings_final_0814_seed1591536269_size10000.csv), specified target dataset(indings_final_0814_seed-53154026_size50.csv).
python dann_synthetic_withTargetLabel.py --lr=0.02 --trade-off=3 --source=findings_final_0814_seed1591536269_size10000 --target=findings_final_0814_seed-53154026_size50
``` 
 
And run the following code to return the corresponding `AUROC` for later performance in the comparison section---all of input data (.csv files of source and target data) are declared in the `if __name__ == '__main__'` module.
```python 
#run program using default hyperparameters---calculate AUC values from all of the models derived from different
python dann_synthetic_withTargetLabel_outputAUC.py`

#run program with specified learning rate (0.02), specified trade-off(3),specified source dataset(findings_final_0814_seed1591536269_size10000.csv), specified target dataset(indings_final_0814_seed-53154026_size50.csv).  
python dann_synthetic_withTargetLabel_outputAUC.py --lr=0.02 --trade-off=3 --source=findings_final_0814_seed1591536269_size10000 --target=findings_final_0814_seed-53154026_size50
``` 
 

(Note: in our code, we have already defined the default hyperparameters -- like *epochs=10, batch_size=32, lr=0.01, momentum=0.9, weight_decay=0.001, trade_off=1.0, etc.*. Those hyperparameters can be changed via [`argparse`](https://docs.python.org/3/library/argparse.html) on command line).

&nbsp;

> ## Model-based Deep Transfer Learning (MDTL) <a name="MDTL"></a>
Model-based transfer learning keeps the source model’s network structure and a few parameters unchanged and tunes the remaining parameters using a few target training data.  We use the following structure for a source model: an input layer, two hidden layers, and an output layer; among these layers, there are three sets of parameters. Thus, there are three model-based transfer learning strategies: tuning all three sets of parameters (**MDTL_Tune_All**), tuning two sets of parameters that involve the two hidden layers and the output layer (**MDTL_Tune2**), and tuning one set of parameters that involves the second hidden layer and the output layer (**MDTL_Tune1**). Because we will compare data-based transfer learning with model-based transfer learning, we choose the same structure as we used in the DANN feature modeling part. That is, two fully connected layers with 128 nodes in each layer was chosen as the hidden layers for the neural network architecture.    
Since the MDTL module relies on the [learned source model](#BL), make sure all the learned source model exists before fine-tuning them.


* **MDTL_Tune1** --- `model-based-TL-TuneLast1Layer.py`    
* **MDTL_Tune2** --- `model-based-TL-TuneLast2Layers.py`    
* **MDTL_Tune_All** --- `model-based-TL-TuneAllLayers.py`
    
For each model, source and target data (.csv file) are fed into models, for training and evaluating. After that, programs will return trained models and save a .txt file recording information of source, target data, accuracy, and AUC during training.

In **MDTL**, we fine-tuned the parameters based on the original source model.    
Based on the structure of Neural Network and the definition of PyTorch, we froze parameters in specefic layers in specific missions.   
For example, in the task of **MDTL_Tune2**, we tuned the parameters in the last two layers; therefore, we froze the parameters in the first layer during the following training under the target setting.  
```python 
# freeze parameters in the first layer.
for param in classifier.fc1.parameters():
	param.requires_grad = False
```


So for the specific tasks, we froze specific parameters, and ran the code below to obtain our models:  
For **MDTL_Tune1**, the parameters of the first two layers are frozen and the souce model is trained under the target setting.
```python 
# run it in the command line, to obtain MDTL_Tune1 model---fine-tuning all of the models derived from different combination of source and target dataset
python model-based-TL-TuneLast1Layer.py

##run program with specified learning rate (0.02), on the model derived from source dataset-findings_final_0814_seed1591536269_size10000.csv, target dataset-indings_final_0814_seed-53154026_size50.csv.  
python model-based-TL-TuneLast1Layer.py --lr=0.02 --source=findings_final_0814_seed1591536269_size10000 --target=findings_final_0814_seed-53154026_size50
```

For **MDTL_Tune2**, the parameters of the first layer are frozen and the source model is trained under the target setting.
```python 
# run it in the command line, to obtain MDTL_Tune2 model---fine-tuning all of the models derived from different combination of source and target dataset
python model-based-TL-TuneLast2Layers.py

##run program with specified learning rate (0.02), on the model derived from source dataset-findings_final_0814_seed1591536269_size10000.csv, target dataset-indings_final_0814_seed-53154026_size50.csv.    
python model-based-TL-TuneLast2Layers.py --lr=0.02 --source=findings_final_0814_seed1591536269_size10000 --target=findings_final_0814_seed-53154026_size50
```

For **MDTL_Tune_ALL**, all of the parameters are fine-tuned under the target setting.
```python 
# run it in the command line, to obtain MDTL_Tune_All model---fine-tuning all of the models derived from different combination of source and target dataset
python model-based-TL-TuneAllLayers.py

##run program with specified learning rate (0.02),on the model derived from source dataset-findings_final_0814_seed1591536269_size10000.csv, target dataset-indings_final_0814_seed-53154026_size50.csv.
python model-based-TL-TuneAllLayers.py --lr=0.02 --source=findings_final_0814_seed1591536269_size10000 --target=findings_final_0814_seed-53154026_size50
```


&nbsp;


> ## Baseline Model <a name="BL"></a>  
Based on the dataset used to train the model, we defined three baseline models. The corresponding codes for each baseline model are shown as below:   
 
+ `learnSourceModel.py` and `learnSourceModel_prob.py`--using the source training dataset to train a model;
+ `learnTargetModel.py` and `learnTargetModel_prob.py`--using the target training dataset to train a model;
+ `learnSourceTargetModel.py` and `learnSourceTargetModel_prob.py` -- using both the source and target training datasets to train a model.

In our code, we have pre-defined some hyperparameters, like  *epochs=10, batch_size=32, lr=0.01, momentum=0.9, weight_decay=0.001, print_freq=100, seed=None, trade_off=1.0, iters_per_epoch=313*, via `argparse`. You can directly change those hyperparameters by using [`argparse`](https://docs.python.org/3/library/argparse.html) through command line.

1. **BL_source**: using the source training dataset to train a model
, and obtaining a trained model under the source setting;
	```pyrhon
	#train different source models based on different souce datasets with default parameters
	python learnSourceModel.py

	#train learnSourceModel on specified source dataset-findings_final_0814_seed1591536269_size10000.csv, learning rate=0.02, epoch=20, initial random seed = 14942.
	python learnSourceModel.py --lr=0.02 --epochs=20 --source=findings_final_0814_seed1591536269_size10000 --seed=14942
	```
	Run `learnSourceModel_prob.py` to get `AUROC` values of BL_source model under the setting.
	```pyrhon
	#calculate AUC of all the learned source model derived from different source datasets
	python learnSourceModel_prob.py

	# get AUC of model derived from source dataset--source=findings_final_0814_seed1591536269_size10000.csv.
	python learnSourceModel_prob.py --source=findings_final_0814_seed1591536269_size10000
	```


2. **BL_target**: using the target training dataset to train a model, and obtaining a trained model under the target setting;
	```pyrhon
	#train different target models based on different target datasets with default parameters
	python learnTargetModel.py

	# train learnTargetModel on specified target dataset-findings_final_0814_seed-53154026_size50.csv, learning rate=0.02, epoch=20, initial random seed = 14942..
	python learnTargetModel.py --target=findings_final_0814_seed-53154026_size50 --lr=0.02 --epochs=20 --seed=14942
	```
	Run `learnTargetModel_prob.py` to get `AUROC` values of BL_target model under the setting.  
	```pyrhon
	#calculate AUC of all the learned target model derived from different target datasets
	python learnTargetModel_prob.py

	# get AUC of model derived from target dataset--target=findings_final_0814_seed-53154026_size50.csv.
	python learnTargetModel_prob.py --target=findings_final_0814_seed-53154026_size50
	```

3. **BL_combined**: using both the source and target training datasets to train a model, and obtaining a trained model under the combined setting 
	```pyrhon
	#using  all of target dataset to train learnSourceTargetModel (with defualt hyperparameters)
	python learnSourceTargetModel.py

	# using target dataset--findings_final_0814_seed-53154026_size50.csv to train learnSourceTargetModel derived from specific source dataset--findings_final_0814_seed1591536269_size10000.csv.   
	python learnSourceTargetModel.py --source=findings_final_0814_seed1591536269_size10000 --target=findings_final_0814_seed-53154026_size50 
	```
	Run `learnSourceTargetModel_prob.py` to get `AUROC` values of BL_combined model under the setting.
	```pyrhon
	#calculate AUC of all the learned SourceTargetModel derived from different target and source datasets
	python learnSourceTargetModel_prob.py

	# get AUC of SourceTargetModel derived from target dataset--target=findings_final_0814_seed-53154026_size50.csv and source dataset--findings_final_0814_seed1591536269_size10000.csv.
	python learnSourceTargetModel_prob.py --source=findings_final_0814_seed1591536269_size10000 --target=findings_final_0814_seed-53154026_size50 
	```

&nbsp;


> ## Main Results and Conclusion <a name="conclusion"></a>

&nbsp; **Our experiments show that simply combining source and target data for modeling does not work well. Both MDTL and DANN perform better than baseline models when the source and target distribution is not largely different (KL is 1), and the target setting has few training samples (size <1000). MDTL models reach a similar performance as DANN models (mean of AUROCs: 0.83 vs. 0.84, P value of Wilcoxon signed-rank test = 0.15). Transfer learning may be useful when the source and target are similar, and the target training data is insufficient. Sharing a well-developed model can be sufficient.**

