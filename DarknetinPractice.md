Instructions for AlexeyAB's darknet framework, simplily noted as darknetv4. YOLOv1-v4, Scaled v4, YOLOv7, and so on are supported.  
There are some differences between Pjreddie's darknetv3 framework.  
just take yolov7x as an example.


## $ git clone

## 1. Prepare your dataset refer to darknetv4/gen_multiobj_dataset_doc.txt
### In AlexeyAB's framework, .txt label files are stored in the same folders (RPdevkit/RP2022/JPEGImages and /JPEGImagesTest) with their corresponding images.

## 2. Before Installing and Compiling darknet - Install your Nvida Driver, CUDA, cuDNN at first
Refer to https://github.com/RealMarco/InstallSoftwaresonUbuntu/blob/master/Install%26CheckDriverCUDAcuDNN
         https://github.com/AlexeyAB/darknet  
         https://github.com/AlexeyAB/darknet/wiki   
and Colab tutorials: 
         https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg   
         https://colab.research.google.com/drive/12QusaaRj_lUwCGDvQNfICpa7kA7_a2dE   
         
As for error and debugging, refer to https://github.com/RealMarco/InstallSoftwaresonUbuntu/blob/master/InstallGPUVersionDarknetv3

### 2.5 install opencv by simply run $ darknetv4/scripts/install_OpenCV4.sh

## 3. Before compiling darknet - Modify *GPU, CUDNN, OPENCV, ARCH* in your Makefile
	GPU=1
	CUDNN=1
	CUDNN_HALF=0
	OPENCV=1
	AVX=0
	OPENMP=0
	LIBSO=0
	ZED_CAMERA=0
	ZED_CAMERA_v2_8=0

	# set GPU=1 and CUDNN=1 to speedup on GPU
	# set CUDNN_HALF=1 to further speedup 3 x times (Mixed-precision on Tensor Cores) GPU: Volta, Xavier, Turing and higher
	# set AVX=1 and OPENMP=1 to speedup on CPU (if error occurs then set AVX=0)
	# set ZED_CAMERA=1 to enable ZED SDK 3.0 and above
	# set ZED_CAMERA_v2_8=1 to enable ZED SDK 2.X

	USE_CPP=0
	DEBUG=0

	ARCH= -gencode arch=compute_35,code=sm_35 \
		  -gencode arch=compute_50,code=[sm_50,compute_50] \
		  -gencode arch=compute_52,code=[sm_52,compute_52] \
			-gencode arch=compute_61,code=[sm_61,compute_61]

	OS := $(shell uname)
	# GeForce RTX 3070, 3080, 3090
	ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]

## 4. Compiling darknet 
$ cd darknet  
$ make clean  # if you have make a CPU version darknet before    
$ make  
### As for error and debugging, refer to https://github.com/RealMarco/InstallSoftwaresonUbuntu/blob/master/InstallGPUVersionDarknetv3

### compiling on windows is quite different from linux, plz refer to AlexeyAB for more details 

## 5. Modify cfg/CASGC.data  
eval = CASGC  
 
## 6. Modify data/CASGC.names in accordance with your classes

## 7. Modify/Check 2022_train.txt and 2022_test.txt, and ensure the text comply with paths of images in RPdevkit/RP2022/JPEGImages and /JPEGImagesTest

## 8. Modify the cfg/yolov7x_CASGC.cfg file  
### look through https://github.com/AlexeyAB/darknet/wiki#meaning-of-configuration-parameters-in-the-cfg-files   for more detials
### width and height should be multiples of 32  
width = 960 # 1920  
height = 512 # 1024  
	### Plz note that when random = 1 in [yolo] heads, the Stochastic Multiscale Training (i.e., width=height, in 320-608, size is changed every 10 epochs) is turned on, and the seetings of width and height is then invalid.  

# recommended to set proper *burn_in*, making burn_in*batch/number_of_training_images  = integer, as well as *max_batches, steps*
burn_in=  
max_batches = 20000 
policy=steps
steps=4200,6400,8000, 8400,8800,9200, 10000,10800,11600  
scales= .5,.5,.4, .5,.5,.4, 2,.5,.1 


### Change the *classes* in the [yolo] heads. Usually no need to adjust mask and num (of anchors)  
classes =   21
ignore_thresh = .7  ### set higher ignore_thresh (of iou) to get more accurate bbx results  
                    ### calculate loss only when IOU(ground-thruth, predicted bbx)>ignore_thresh

### calculate anchors, then copy them into *anchors* in [yolo] heads  
$ ./darknet detector calc_anchors cfg/CASGC.data -num_of_clusters 9 -width 960 -height 512 
Or $  python gen_anchors.py  

### change the *filters* in the last [convolutional] layers before [yolo] heads   
filters =<number of mask>*(classes + 5) ### 78 = 3*(21+5)  

### Adjust the parameters of data augmentation when training according to your data (directly use default settings without adjustment is also ok)  
	angle=0 # or try angle=180
	saturation = 1.3
	exposure = 1.5 # 1/1.5 - 1.5, deal with abnormal lighting conditions
	hue=.1  # -0.1 - +0.1
	mosaic =1
	jitter=.1
	#random =1 
	Or random =0


## 7. Add a cfg/yolov7x_CASGC_test.cfg file for testing and evaluation 


## 8. Training 
$ ./darknet detector train cfg/CASGC.data cfg/yolov7x_CASGC.cfg -map -iou_thresh 0.7 2>1 |tee visualization/train_yolov7x_CASGC_iou.7.log
	### -map  ,validating every 4 epochs when training 
	### -iou_thresh  ,default =.5, for evaluation (calcuating mAP@iou_thresh) only, nothing with training
	### -thresh  confidence threshold for calculate precision, recall, F1-score, average IoU when validating, adjust the thresh according to the evaluation results of these metrics
	### set a higher -iou_thresh to improve the metric mAP@iou_thresh, which indicates the accuracy of bbx.
	$ ./darknet detector train cfg/CASGC.data cfg/yolov7x_CASGC.cfg backup/your_best.weights  -map -iou_thresh 0.8 -thresh .75 2>1 |tee visualization/train_yolov7x_CASGC_iou.8.log
		### change parameters below in the [yolo] heads, it will increase mAP@0.8, but decrease mAP@0.5
		ignore_thresh = .8 
		iou_normalizer=0.5 
		iou_loss=giou 
	$ ./darknet detector train cfg/CASGC.data cfg/yolov7x_CASGC.cfg backup/your_best.weights -map -iou_thresh 0.83 -thresh .62 2>1 |tee visualization/train_yolov7x_CASGC_iou.83.log

>>> darkent.exe xxx for windows

### Copy and rename the chart_xxx.png, .log and xxx_last.weights once pausing the training, in order to backup them.


## 9. Checkpoint Training/ Transfer Learning: sepcify pretrained weights and ensure that *max_batches* is bigger than trained batches.  
$ ./darknet detector train cfg/CASGC.data cfg/yolov7x_CASGC.cfg backup/yolo-voc_final.weights  
	### P.S. The trained batches should be smaller than the param “max_batches” in .cfg file. Otherwise, the program would recognize it as training end point. At that time, you could also continue training by modifying the max_batches in .cfg file.  


## 10. Evaluation and Visualization
$ ./darknet detector map cfg/CASGC.data cfg/yolov7x_CASGC_test.cfg backup/yolov7x_CASGC_iou_11742.weights  -iou_thresh 0.75 -thresh .75 
	### -iou_thresh .75: calculate mAP@.75
	### -thresh: confidence threshold for calculate precision, recall, F1-score, average IoU when validating 
	### -points ?
	### ./darknet detector map cfg/CASGC.data cfg/yolov7x_CASGC_test_iou.8.cfg backup/yolov4-p5/yolov7x_CASGC_iou_15390.weights -iou_thresh 0.8 -thresh .62

$ ./darknet detector recall cfg/CASGC.data cfg/yolov7x_CASGC_test.cfg backup/yolov7x_CASGC_iou_11742.weights 
    ### Observing the IoU distribution on each category of validation set 
    ### -iou_thresh 0.75

$ ./darknet detector test cfg/CASGC.data cfg/yolov7x_CASGC_test.cfg backup/yolov7x_CASGC_best_11_7.weights RPdevkit/RP2022/test/0010r.jpg 
    ### test a batch of images and visualize/show the original images with bbxes; The .txt file should be in a specific format, viz. one path to an image per line.
    ./darknet detector test cfg/CASGC.data cfg/yolov7x_CASGC_test.cfg backup/yolov7x_CASGC_best_11_7.weights RPdevkit/RP2022/test/images.txt 
    ### set -thresh .75
    ### save results to .json file
    ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -ext_output -dont_show -out result.json < /mydrive/images.txt
    ### save results to .txt file
    ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show -ext_output < /mydrive/images.txt > result.txt
    

### Visualize loss-batches and iou-batchs by .log in /visualization folder
$ cd visualization
$ python extract_log.py
$ python train_loss_visualization.py
$ python train_iou_visualization.py
$ cd ..

### visualize PR curve
#### Generate predicted bbx and confidence score in results/comp4_det_test_shoe.txt
$ ./darknet detector valid cfg/CASGC.data cfg/yolov7x_CASGC_test.cfg backup/yolov4-p5/yolov7x_CASGC_iou_16986.weights  -thresh .4	
#### generate testRP/shoe_pr.pkl 
$ python reval_RP.py  --RP_dir /home/marco/darknet/RPdevkit --year 2022 --image_set test --classes  /home/marco/darknet/data/CASGC.names  /home/marco/darknet/testRP
#### modify the *fr = open('testRP/shoe_pr.pkl','rb')* line in draw_pr.py, and then run it
$ python draw_pr.py

## 11. Source Code Finetuning and Recompling
### metric score = 0.34IOU + 0.33Class + 0.33Obj, 
### Saving models with best metric score, and saving model after every epoch when finetuning


## 12. Special tips for detecting small objects
1) Train several times, and set a higher *ignore_thresh* in [yolo] heads in .cfg gradually, then change the corresponding -iou_thresh when run $ ./darknet detector train ...  
	### change parameters below in the [yolo] heads, it will increase mAP@0.8, but decrease mAP@0.5
	ignore_thresh = .8 
	iou_normalizer=0.5 
	iou_loss=giou 
	
2) use input images in higher resolution, and change width and height in .cfg and always ensure that random =0

3) Observing the IoU distribution on each category of validation set, and try to improve the performance of categories with lower scores accordingly

4) Also try yolov4-custom.cfg and yolov4-p5 (896x896), yolov4-csp-x-swish (640x640), yolov4-p6(1280x1280)


## 13. Special tricks for detecting objects in abnormal light conditions
exposure = 1.5
jitter=.1


# If you are using YOLOv5 in PyTorch version,

## 1. When training on multi-GPUs, set DP (DataParallel) mode for singe-machine multi-GPU training, or DDP (DistributeDataParallel) mode for multi-machine multi-GPU training. For instance  

model = torch.nn.DataParallel(model)
and handle on other parameters and errors.

Besides, you can use Model Parallel method.







































































































































































































