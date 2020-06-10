#!/usr/bin/env bash


wget -i "splits/$1_to_download.txt" -P "data/$1/"

$argo = 'argoverse'

cd "data/$1"

if [ $1 == $argo]
then
	tar -xvzf "*.tar.gz"
else
	unzip "*.zip"
fi

cd ../../

