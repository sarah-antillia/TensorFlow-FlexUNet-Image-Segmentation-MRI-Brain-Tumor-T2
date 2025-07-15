<h2>TensorFlow-FlexUNet-Image-Segmentation-MRI-Brain-Tumor-T2 (2025/07/16)</h2>

This is the first experiment of Image Segmentation for MRI-Brain-Tumor-T2 based on our TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1Gf5Osuuy1X-FWL-zM4AmfusQY6TCZEih/view?usp=sharing">
Brain-PNG-MRI-T2-ImageMask-Dataset.zip</a>.
which was derived by us from <br>
<a href="https://data.mendeley.com/datasets/8bctsm8jz7/1">
Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information 
</a>
<br><br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a> ,
 our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br>
<br>
<b>Acutual Image Segmentation for 512x512 MRI-Brain-Tumor-T2 images</b><br>
As shown below, the inferred masks look very similar to the ground truth masks. 
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/images/13223.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/masks/13223.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test_output/13223.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/images/13407.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/masks/13407.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test_output/13407.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/images/13521.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/masks/13521.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test_output/13521.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
We used the following dataset to create our ImageMask dataset <br>
<a href="https://data.mendeley.com/datasets/8bctsm8jz7/1"><b>
Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information 
</b>
</a>
<br>
<br>
Published: 31 March 2022|Version 1|DOI:10.17632/c
<br>
<br>
<b>Contributor:</b>Ali M Muslim
<br>
<br>
<b>Description:</b><br>
Magnetic resonance imaging (MRI) provides a significant key to diagnose and monitor the progression of Multiple Sclerosis (MS) disease. Manual MS-Lesion segmentation, Expanded Disability Status Scale (EDSS) and patient’s meta information can provide a gold standard for research in terms of automated MS-lesion quantification, automated EDSS prediction and identification of the correlation between MS-lesion and patient disability. In this dataset, we provide a novel multi-sequence MRI dataset of 60 MS patients with consensus manual lesion segmentation, EDSS, general patient information and clinical information. On this dataset, three radiologists and neurologist experts segmented and validated the manual MS-lesion segmentation for three MRI sequences T1-weighted, T2-weighted and fluid-attenuated inversion recovery (FLAIR). The dataset can be used to study the relationship between MS-lesion, EDSS and patient clinical information. Furthermore, it also can be used to development of automated MS-lesion segmentation, patient disability prediction using MRI and correlation analysis between patient disability and MRI brain abnormalities include MS lesion location, size, number and type. 
<br>
<br>
<b>Licence:</b> CC BY 4.0<br>
<br>
<br>

<h3>
<a id="2">
2 MRI-Brain-Tumor-T2 ImageMask Dataset
</a>
</h3>
 If you would like to train this MRI-Brain-Tumor-T2 Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1Gf5Osuuy1X-FWL-zM4AmfusQY6TCZEih/view?usp=sharing">
Brain-PNG-MRI-T2-ImageMask-Dataset.zip</a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─MRI-Brain-Tumor-T2
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>MRI-Brain-Tumor-T2 Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/MRI-Brain-Tumor-T2_Statistics.png" width="512" height="auto"><br>
<br>
On the derivation of the dataset, please refer to our repository:<br>
<a href="https://github.com/sarah-antillia/ImageMask-Dataset-Multiple-Sclerosis-Brain-MRI">
ImageMask-Dataset-Multiple-Sclerosis-Brain-MRI
</a>
<br><br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained MRI-Brain-Tumor-T2 TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2 and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00005
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for MRI-Brain-Tumor-T2 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
; background:black  tumor: white
rgb_map = {(0,0,0):0, (255,255,255):1 }
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 21,22,23)</b><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 44,45,46)</b><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 46 by EarlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/asset/train_console_output_at_epoch46.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for MRI-Brain-Tumor-T2.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/asset/evaluate_console_output_at_epoch46.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this MRI-Brain-Tumor-T2/test was very low and dice_coef_multiclass 
very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0101
dice_coef_multiclass,0.9949
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for MRI-Brain-Tumor-T2.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/images/13223.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/masks/13223.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test_output/13223.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/images/13407.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/masks/13407.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test_output/13407.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/images/13521.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/masks/13521.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test_output/13521.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/images/13920.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/masks/13920.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test_output/13920.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/images/14215.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/masks/14215.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test_output/14215.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/images/15413.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test/masks/15413.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/MRI-Brain-Tumor-T2/mini_test_output/15413.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Consensus of algorithms for lesion segmentation in brain MRI studies of multiple sclerosis </b><br>
Alessandro Pasquale De Rosa, Marco Benedetto, Stefano Tagliaferri, Francesco Bardozzo, Alessandro D’Ambrosio, <br>
Alvino Bisecco, Antonio Gallo, Mario Cirillo, Roberto Tagliaferri & Fabrizio Esposito <br>

<a href="https://www.nature.com/articles/s41598-024-72649-9">https://www.nature.com/articles/s41598-024-72649-9</a>
<br>
<br>
<b>2. Tensorflow-Image-Segmentation-Multiple-Sclerosis-Brain-MRI-T1</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Multiple-Sclerosis-Brain-MRI-T1">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Multiple-Sclerosis-Brain-MRI-T1
</a>
<br>
<br>
<b>3. Tensorflow-Image-Segmentation-Multiple-Sclerosis-Brain-MRI-Flair</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Multiple-Sclerosis-Brain-MRI-Flair">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Multiple-Sclerosis-Brain-MRI-Flair
</a>
