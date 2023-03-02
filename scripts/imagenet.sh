DATA=coop_data/
cd $DATA

#mkdir -p imagenet/images/
cd imagenet/images/

# 1. Download the data
#get ILSVRC2012_img_val.tar (about 6.3 GB). MD5: 29b22e2961454d5413ddabcf34fc5622
#wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
#get ILSVRC2012_img_train.tar (about 138 GB). MD5: 1d675b47d978889d74fa0da5fadfb00e
#wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

## 2. Extract the training data:
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && mv ILSVRC2012_img_train.tar coop_data/imagenet/
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

# ## 3. Extract the validation data and move images to subfolders:
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar && mv ILSVRC2012_img_val.tar coop_data/imagenet/
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash


## 4. Move the classname.txt to /imagenet/images
cd ../../../../scripts/
mv classnames.txt ../coop_data/imagenet/images