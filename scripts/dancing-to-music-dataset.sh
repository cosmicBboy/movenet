#!/bin/sh

# Script adapted from https://vllab.ucmerced.edu/hylee/Dancing2Music/script.txt

#### 3 zip files containing data of three dancing categories: Zumba, ballet, and hiphop.
#### 1 zip files containing data statistics and data path lists for training usage.

root_path=./datasets
data_path=$root_path/dancing_to_music
root_url=http://vllab.ucmerced.edu/hylee/Dancing2Music

mkdir -p $data_path

download_data() {
    file=$1
    wget -N $root_url/$file -O $data_path/$file
    unzip $data_path/$file -d $data_path
    rm $data_path/$file
}

for file in ballet.zip zumba.zip hiphop.zip data.zip
do
    download_data $file
done;
