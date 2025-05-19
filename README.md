# **GeoDML: Minimizing Shape Distortions in Geometric Deep Neural Network for Skeleton-Based Human Rehabilitation Exercise and Disease Assessments** 

## Abstract
<div style="text-align: justify"> 
Nowadays, geometric deep learning is gaining lots of attraction in the deep learning domain due to their meaningful representation of input data, especially those lying in a non-euclidean space. However, the process of geometrically representing these data in a convenient form for deep learning layers usually introduces distortions; for example, the loss of information when approximating the processed input data from the manifold to the linear tangent space. To address this issue and minimize these distortions, we propose a novel geometric distortion minimization approach (denoted GeoDML) to reduce the approximation errors that occur when transferring these data to the linear tangent space. To demonstrate the effectiveness and efficiency of our GeoDML, we incorporate it into a state-of-the-art geometric deep learning framework and evaluate its performance on three public and real-world human skeleton-based disease and rehabilitation datasets. With our proposed approach, the performance of the network improves greatly from baseline especially in challenging scenarios while also achieving state-of-the-art performance across these datasets. Additionally, our proposed GeoDML comes with a negligible computational cost and we conduct an ablation study on its various variants to better understand and highlight the best one for minimizing shape distortions in the geometric network.
</div>

## Packages and Dependencies
- For packages and dependencies, first create an enviroment using Python, activate the enviroment and run `pip install -r requirements.txt` . We run all our experiments on Python version 3.12.6

## Datasets
- All the three datasets that we have used in our work can be obtained from [here](https://github.com/bruceyo/EGCN/tree/master) 

## Scripts Organization
- Each dataset folder contains all the scripts for the respective dataset and category.

## Usage or Training and Evaluation Per Dataset and Exercise 
- To run our model for EHE, KIMORE or UI-PRMD per exercise, `cd` into the 'run_files' folder and then  `python .\run_file_GeoDML_Softmax.py` or the Sigmoid file for our best variants described in the paper. You can similarly run `python .\run_file_KShapeNet_softmax.py` or the Sigmoid version for KShapeNet only.
  
## Test Probabilities Evaluation Metrics
- To compute ED, CR, SD and DM, use the file 'compute_euclid_and_corr' in the EHE forlder for EHE dataset; the file 'compute_cr_corr_KIMORE' for the KIMORE dataset; and the file 'compute_sd_and_dm_UI-PRMD' for the UI-PRMD dataset.

## Usage or Training and Evaluation Per Dataset for Multi-Class
- To perform multi-class training and evaluation, use the folder for each dataset with the name 'multi_class' appeneded. `cd` into the 'run_files' folder and `python .\GeoDML.py` for ours or `python .\Kshape_only.py` for KShapeNet. 

## Evaluation on NTU-RDB+D and NTU-RGB+D120 Datasets
- Download the dataset (the skeleton files only are sufficient) from the dataset page [here](https://rose1.ntu.edu.sg/dataset/actionRecognition/)

**1- Organization:**
Organize the data folder to look like this.

	├───NTU-RGB+D
	│   └───nturgb+d_skeletons /* .skeleton files for NTU60 under this folder
	├───NTU-RGB+D120
	│   └───nturgb+d_skeletons /* .skeleton files for NTU120 under this folder
	├───nturgb_d
	│   ├───xsub
	│   └───xview
	└───nturgb_d120
		├───xsetup
		└───xsub
  
**2- Data generation from .skeleton files**
Generate the main numpy files used for the data generation (This code goes to the creators of the [AS-GCN](https://github.com/limaosen0/AS-GCN) network). **Please make sure the data conforms to the previously presented tree** <br>
run `cd data_gen` , then `python ntu_gen_preprocess60.py` for NTU-60 or `python ntu_gen_preprocess120.py` for NTU-120. 
Result example for NTU60 xview should look like this.

	│   └───xview
	│           .gitkeep
	│           train_data_joint_pad.npy
	│           train_label.pkl
	│           val_data_joint_pad.npy
	│           val_label.pkl
 
**3- Data generation from 'joint_pad.npy files', make sure you're still in the data_gen folder** <br>
run `python generate_2bodies.py` for NTU-60 or `python generate_2bodies_120.py` for NTU-120. 
Result example after generation for NTU60 xview should appear like this:

    │   └───xview
    │           .gitkeep
    │           test_xview_interp100.npy
    │           train_data_joint_pad.npy
    │           train_label.pkl
    │           train_xview_interp100.npy
    │           val_data_joint_pad.npy
    │           val_label.pkl

**4- Model training and testing, (please make sure you're in the root file of the repository)**
run the file `NTU60_KShape.py` for KShapeNet on NTU-60 and the file `NTU60_GeoDML.py` for GeoDML on NTU-60. Run the other corresponding scripts for NTU-120. 
--prot [xview,xsub] --num_epoch [EPOCHS] --layer_name [RigidTransform, RigidTransformInit, NonRigidTransform, NonRigidTransformInit] --learning_rate [lr] --batch_size [size] --batch_size_test [size'] --save [True,False]
