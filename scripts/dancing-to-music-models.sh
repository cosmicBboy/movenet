#!/bin/sh

# Script adapted from https://vllab.ucmerced.edu/hylee/Dancing2Music/script.txt

model_path=./models/dancing_to_music
mkdir -p $model_path

wget -N http://vllab.ucmerced.edu/hylee/Dancing2Music/Stage1.ckpt -O $model_path/Stage1.ckpt
wget -N http://vllab.ucmerced.edu/hylee/Dancing2Music/Stage2.ckpt -O $model_path/Stage2.ckpt
wget -N https://www.dropbox.com/s/1c7s7rn7z3pvhp9/Model_MY.zip -O .$model_path/Model_MY.zip
