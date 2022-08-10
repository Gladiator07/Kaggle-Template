#!/bin/bash
<<com
Supports colab, kaggle and jarvislabs environment setup
Usage:
bash setup.sh <ENVIRON> <download_data_or_not>
Example:
bash setup.sh colab true
com

ENVIRON=$1
DOWNLOAD_DATA=$2
PROJECT=""

pip install --upgrade -r requirements.txt

cd ~/.

if [ "$DOWNLOAD_DATA" == "true" ]
then
    echo "Downloading data"
    if [ "$1" == "colab" ]
    then
        cd /content/$PROJECT
    elif [ "$1" == "jarvislabs" ]
    then
        cd /home/$PROJECT
    else
        echo "Unrecognized environment"
    fi
else
    echo "Data download disabled"
fi

mkdir input/
cd input/

# download competition data

# download other data