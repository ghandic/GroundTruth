### -- Setting up a TF model on Google Cloud -- ###

# Download Google Cloud SDK
cd
wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-177.0.0-darwin-x86_64.tar.gz
tar -xvf google-cloud-sdk-177.0.0-darwin-x86_64.tar.gz
./google-cloud-sdk/install.sh


## You may need to restart the terminal

## Model selection and data

export GROUND_TRUTH_REPO_DIR="${HOME}/Documents/GitHub/GroundTruth/"
export DATA_DIR="${GROUND_TRUTH_REPO_DIR}TensorFlow/data/"
export IMG_DIR="${GROUND_TRUTH_REPO_DIR}Images/"
export CONFIG_DIR="${GROUND_TRUTH_REPO_DIR}TensorFlow/configs/"

export MODEL_TYPE="ssd_mobilenet" # or faster_rcnn

export DATA_IN="${DATA_DIR}bokeh_result.csv"
export OBJECT_MAP="${DATA_DIR}object-detection.pbtxt"
export TRAIN_PER=0.75

cat >$OBJECT_MAP <<EOL
item {
  id: 1
  name: 'cat'
}
EOL

## Setting up the environment
gcloud auth application-default login
virtualenv --system-site-packages ~/tensorflow
source ~/tensorflow/bin/activate
sudo pip install tensorflow pillow lxml pandas

export PATH=~/google-cloud-sdk/bin:~/tensorflow/bin:$PATH
export PROJECT=$(gcloud config list project --format "value(core.project)")
export YOUR_GCS_BUCKET="gs://${PROJECT}-ml"

git clone https://github.com/tensorflow/models
cd models/research
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" < /dev/null 2> /dev/null
brew install protobuf
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

#### Test it works
python object_detection/builders/model_builder_test.py

## Creating protocol buffer record files and object map (setting all the data)
# Create a map of labels

python ${GROUND_TRUTH_REPO_DIR}/TensorFlow/generate_clean_csv.py --csv_input=${DATA_IN} --img_dir=$IMG_DIR --output_path=${DATA_DIR} --train_percent=${TRAIN_PER}
python ${GROUND_TRUTH_REPO_DIR}/TensorFlow/generate_tfrecord.py --csv_input=${DATA_DIR}/train_records.csv  --output_path=${DATA_DIR}train.record --pbtxt=${OBJECT_MAP} --img_dir=${IMG_DIR}
python ${GROUND_TRUTH_REPO_DIR}/TensorFlow/generate_tfrecord.py --csv_input=${DATA_DIR}/test_records.csv  --output_path=${DATA_DIR}test.record --pbtxt=${OBJECT_MAP} --img_dir=${IMG_DIR}

## Copy all three of these to Google cloud
gsutil cp ${DATA_DIR}train.record ${YOUR_GCS_BUCKET}/data/train.record
gsutil cp ${DATA_DIR}test.record ${YOUR_GCS_BUCKET}/data/test.record
gsutil cp ${OBJECT_MAP} ${YOUR_GCS_BUCKET}/data/object-detection.pbtxt

## Getting a model and making config
## ENSURE YOU ARE IN models/research at this point
## ---------------------------------
if [ $MODEL_TYPE == "faster_rcnn" ]
then
	## Getting a model
	wget -N https://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
	tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
	gsutil cp faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.* ${YOUR_GCS_BUCKET}/data/

	## Configure the model run
	cp ${CONFIG_DIR}faster_rcnn_resnet101.config ${CONFIG_DIR}faster_rcnn_resnet101_retrained.config
	sed -i ".original" "s|PATH_TO_BE_CONFIGURED|"${YOUR_GCS_BUCKET}"/data|g" ${CONFIG_DIR}faster_rcnn_resnet101_retrained.config
	rm ${CONFIG_DIR}faster_rcnn_resnet101_retrained.config.original
	gsutil cp ${CONFIG_DIR}/faster_rcnn_resnet101_retrained.config ${YOUR_GCS_BUCKET}/data/faster_rcnn_resnet101_retrained.config

	gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
	    --job-dir=${YOUR_GCS_BUCKET}/train \
	    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
	    --module-name object_detection.train \
	    --region us-central1 \
	    --config object_detection/samples/cloud/cloud.yml \
	    -- \
	    --train_dir=${YOUR_GCS_BUCKET}/train \
	    --pipeline_config_path=${YOUR_GCS_BUCKET}/data/faster_rcnn_resnet101_retrained.config
 
	gcloud ml-engine jobs submit training `whoami`_object_detection_eval_`date +%s` \
	    --job-dir=${YOUR_GCS_BUCKET}/train \
	    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
	    --module-name object_detection.eval \
	    --region us-central1 \
	    --scale-tier BASIC_GPU \
	    -- \
	    --checkpoint_dir=${YOUR_GCS_BUCKET}/train \
	    --eval_dir=${YOUR_GCS_BUCKET}/eval \
	    --pipeline_config_path=${YOUR_GCS_BUCKET}/data/faster_rcnn_resnet101_retrained.config

	tensorboard --logdir=${YOUR_GCS_BUCKET}
	
elif [ $MODEL_TYPE == "ssd_mobilenet" ]
then
	## Getting a model
	wget -N http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
	tar -xvf ssd_mobilenet_v1_coco_11_06_2017.tar.gz
	gsutil cp ssd_mobilenet_v1_coco_11_06_2017/model.ckpt.* ${YOUR_GCS_BUCKET}/data/

	## Configure the model run
	cp ${CONFIG_DIR}ssd_mobilenet_v1.config ${CONFIG_DIR}ssd_mobilenet_v1_retrained.config
	sed -i ".original" "s|PATH_TO_BE_CONFIGURED|"${YOUR_GCS_BUCKET}"/data|g" ${CONFIG_DIR}ssd_mobilenet_v1_retrained.config
	rm ${CONFIG_DIR}ssd_mobilenet_v1_retrained.config.original
	gsutil cp ${CONFIG_DIR}ssd_mobilenet_v1_retrained.config ${YOUR_GCS_BUCKET}/data/ssd_mobilenet_v1_retrained.config

	gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
	    --job-dir=${YOUR_GCS_BUCKET}/train \
	    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
	    --module-name object_detection.train \
	    --region us-central1 \
	    --config object_detection/samples/cloud/cloud.yml \
	    -- \
	    --train_dir=${YOUR_GCS_BUCKET}/train \
	    --pipeline_config_path=${YOUR_GCS_BUCKET}/data/ssd_mobilenet_v1_retrained.config
 
	gcloud ml-engine jobs submit training `whoami`_object_detection_eval_`date +%s` \
	    --job-dir=${YOUR_GCS_BUCKET}/train \
	    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
	    --module-name object_detection.eval \
	    --region us-central1 \
	    --scale-tier BASIC_GPU \
	    -- \
	    --checkpoint_dir=${YOUR_GCS_BUCKET}/train \
	    --eval_dir=${YOUR_GCS_BUCKET}/eval \
	    --pipeline_config_path=${YOUR_GCS_BUCKET}/data/ssd_mobilenet_v1_retrained.config

    tensorboard --logdir=${YOUR_GCS_BUCKET}

else
	echo "you have not selected a model"
fi



echo "Please define a checkpoint number based on the checkpoint you’d like to export"
read CHECKPOINT_NUMBER

gsutil cp ${YOUR_GCS_BUCKET}/train/model.ckpt-${CHECKPOINT_NUMBER}.* .

if [ $MODEL_TYPE == "faster_rcnn" ]
then
	python object_detection/export_inference_graph.py \
	    --input_type image_tensor \
	    --pipeline_config_path ${CONFIG_DIR}faster_rcnn_resnet101_retrained.config \
	    --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
	    --output_directory ${GROUND_TRUTH_REPO_DIR}TensorFlow/models/faster_rcnn
elif [ $MODEL_TYPE == "ssd_mobilenet" ]
then
	python object_detection/export_inference_graph.py \
	    --input_type image_tensor \
	    --pipeline_config_path ${CONFIG_DIR}ssd_mobilenet_v1_retrained.config \
	    --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
	    --output_directory ${GROUND_TRUTH_REPO_DIR}TensorFlow/models/ssd_mobilenet
else
	echo "you have not selected a model"
fi



