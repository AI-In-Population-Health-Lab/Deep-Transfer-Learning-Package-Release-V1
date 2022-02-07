# AI_public_lab 
**Machine learning is a powerful tool in the biomedical field; however, at times machine learning suffers from the deficiency in the quantity and quality of the training data. To overcome these limitations of machine learning due to the limitations of data, transfer learning has been introduced.**  
**In this paper, we explore the deep transfer learning (DTL) methods for infectious disease case detection. This method uses shared EHR raw data or models building from the source data to predict infectious disease pandemic cases in a target region.**
&nbsp;
&nbsp;

## configuration  
- [x] Python (version--3.9)
- [x] PyTorch (version--1.10.1) 
- [x] torchvision (version--0.11.3)

All you can install via `conda` or `pip`.


&nbsp;

----

**In the following section, we try to use this README.md file to explain how our code works, hopefully making our project easier to understand for viewers, who might want to leverage the project for further research.**

----


> # Data processing  
In the transfer learning field, the divergence between source and target data is the essence of this kind of task. Our jobs are trying to study an effective methodology of transfer learning in the biomedical field, leveraging this fascinating technique to empower  infection detecting algorithms via electronic medical records (EHRs).   
Because source and target data are recorded/presented differently, the data processing is a fundational and essential step of this project

In our project, we get source and target data with different levels of divergence --- Kullback–Leibler divergence.    
The definition of KL-divergence is shown below(*t*  represents target, *s* represents source ).
![image](https://user-images.githubusercontent.com/39432361/152086546-b42438da-7f0e-411a-b08f-89cc445af061.png)

Data in the source setting may have/often has different dimensions than data extracted from the target setting, so before training our model, we must make sure, the source and target data can be mapped into the same space. We solve this problem by **padding extra zeros** so that data from different settings have the same dimensionality. 

After this recalibration of the data, we use all of the source data for model training, and split our target data into different portions, for training, validation, or testing in the following models.    
&nbsp;






> # Data-based Deep Transfer Learning (DDTL) 
**In DDTL section,  we apply the Domain Adversarial Neural Networks[(DANN)](https://arxiv.org/abs/1505.07818)**.  
DANN works in three scenarios: the target training data have class labels (DANN_supervised); the target training data do not have any class label (DANN_unsupervised); a portion of the target training data have class labels (DANN_semisupervised). We will assess the first two in our experiments.    
&nbsp;    
For Domain Adversarial Neural Networks(DANN), the *Loss function* is defined as below:
![image](https://user-images.githubusercontent.com/39432361/152067006-54cb0fef-d557-47f0-81eb-de2f3ddc39d4.png)
&nbsp;

* **DANN(unsupervised)**---`dann_synthetic_noTargetLabel_noTargetVal.py`, `dann_synthetic_noTargetLabel_noTargetVal_outputAUC.py`

	In the unsupervised DANN, Label Classifier is only trained by sourse training data (target training data is not to be used for the training of Label Classifier). Both target and source data are to be used for training Feature Generator and Domain Classifier. 

	In our project, we run the following code to train the DDTL model under an unsupervised setting.  
	```python 
	python dann_synthetic_noTargetLabel_noTargetVal.py
	``` 

	And run the following code to return the corresponding `AUROC` for later performance in the comparison section. 
	```python 
	python dann_synthetic_noTargetLabel_noTargetVal_outputAUC.py
	``` 


&nbsp;

* **DANN (supervised)**---`dann_synthetic_withTargetLabel.py`, `dann_synthetic_withTargetLabel_outputAUC.py`

	 In the supervised DANN, target training data is be used for the training of Label Classifier, as well as sourse training data. Both target and source data are to be used for training Feature Generator and Domain Classifier. 

	 In our project, we run the following code to train the DDTL model under an unsupervised setting,
	```python 
	python dann_synthetic_withTargetLabel.py
	``` 
	 
	And run the following code to return the corresponding `AUROC` for later performance in the comparison section.
	```python 
	python dann_synthetic_withTargetLabel_outputAUC.py`
	``` 
 



(Note: in our code, we have already defined the default hyperparameters -- like *epochs=10, batch_size=32, lr=0.01, momentum=0.9, weight_decay=0.001, trade_off=1.0, etc.*. Those hyperparameters can be changed via [`argparse`](https://docs.python.org/3/library/argparse.html) on command line).



	

&nbsp;
&nbsp;

> # Model-based Deep Transfer Learning (MDTL) 
Model-based transfer learning keeps the source model’s network structure and a few parameters unchanged and tunes the remaining parameters using a few target training data.  We use the following structure for a source model: an input layer, two hidden layers, and an output layer; among these layers, there are three sets of parameters. Thus, there are three model-based transfer learning strategies: tuning all three sets of parameters (**MDTL_Tune_All**), tuning two sets of parameters that involve the two hidden layers and the output layer (**MDTL_Tune2**), and tuning one set of parameters that involves the second hidden layer and the output layer (**MDTL_Tune1**). Because we will compare data-based transfer learning with model-based transfer learning, we choose the same structure as we used in the DANN feature modeling part. That is, two fully connected layers with 128 nodes in each layer was chosen as the hidden layers for the neural network architecture.


* **MDTL_Tune1** --- `model-based-TL-TuneLast1Layer.py`    
* **MDTL_Tune2** --- `model-based-TL-TuneLast2Layers.py`    
* **MDTL_Tune_All** --- `model-based-TL-TuneAllLayers.py`
    

In **MDTL**, as the description in the paper, we fine-tuned the parameters based on the original source model.    
Based on the structure of Neural Network and the definition of PyTorch, we froze parameters in specefic layers in specific missions.   
For example, in the task of **MDTL_Tune2**, we  tuned the parameters in the last two layers; therefore, we froze the parameters in the first layer during the following training under the target setting.  
```python 
# freeze parameters in the first layer.
for param in classifier.fc1.parameters():
	param.requires_grad = False
```


So for the specific tasks, we froze specific parameters, and ran the code below to obtain our models:  
For **MDTL_Tune1**, the parameters of the first two layers are frozen and the souce model is trained under the target setting.
```python 
# run it in the command line, to obtain MDTL_Tune1 model
python model-based-TL-TuneLast1Layer.py
```

For **MDTL_Tune2**, the parameters of the first layer are frozen and the source model is trained under the target setting.
```python 
# run it in the command line, to obtain MDTL_Tune2 model
python model-based-TL-TuneLast2Layer.py
```

For **MDTL_Tune_ALL**, all of the parameters are be fine-tuned under the target setting.
```python 
# run it in the command line, to obtain MDTL_Tune_All model
python model-based-TL-TuneAllLayers.py
```

(Note: in our code, we have already defined the default hyperparameters -- like *epochs=10, batch_size=32, lr=0.01, momentum=0.9, etc.*. Those hyperparameters can be changed via [`argparse`](https://docs.python.org/3/library/argparse.html) on command line.)


&nbsp;


> # Baseline Model   
Based on the dataset used to train the model, we define three baseline models. The corresponding codes for each baseline model are shown as below:   
 
+ `learnSourceModel.py` and `learnSourceModel_prob.py`--using the source training dataset to train a model;
+ `learnTargetModel.py` and `learnTargetModel_prob.py`--using the target training dataset to train a model;
+ `learnSourceTargetModel.py` and `learnSourceTargetModel_prob.py` -- using both the source and target training datasets to train a model.

In our code, we have predefined some hyperparameters, like  *epochs=10, batch_size=32, lr=0.01, momentum=0.9, weight_decay=0.001, print_freq=100, seed=None, trade_off=1.0, iters_per_epoch=313*, via `argparse`. You can directly change those hyperparameters by using [`argparse`](https://docs.python.org/3/library/argparse.html) through command line.

By default setting, directly run the following codes through the command line:

1. **BL_source**: using the source training dataset to train a model
, and obtaining a trained model under the source setting;
	```pyrhon
	python learnSourceModel.py
	```
	Run `learnSourceModel_prob.py` to get `AUROC` values of BL_source model under the setting.
	```pyrhon
	python learnSourceModel_prob.py
	```


2. **BL_target**: using the target training dataset to train a model, and obtaining a trained model under the target setting;
	```pyrhon
	python learnTargetModel.py
	```
	Run `learnTargetModel_prob.py` to get `AUROC` values of BL_target model under the setting.  
	```pyrhon
	python learnTargetModel_prob.py
	```

3. **BL_combined**: using both the source and target training datasets to train a model, and obtaining a trained model under the combined setting 
	```pyrhon
	python  learnSourceTargetModel.py
	```
	Run `learnSourceTargetModel_prob.py` to get `AUROC` values of BL_combined model under the setting.
	```pyrhon
	python learnSourceTargetModel_prob.py
	```


&nbsp;


> # Conclusion 

&nbsp; **Our experiments show that in two situations, in the context of infectious disease detection, DTL can be useful when *(1) the target training data is insufficient* and *(2) the target training data does not have labels.*        
&nbsp; MDTL and DANN_supervised methods work well in the first situation, especially when the dissimilarity between the source and the target is large.       
&nbsp; DANN_unsupervised works well in the second situation, especially when the dissimilarity between the source and the target is large while the target features are limited.**


## Code will be released soon
