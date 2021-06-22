#!/bin/bash

sname=$1"/"
fdir="/mnt/zfsusers/mabitbol/BBPower/moments/"
ndir=$fdir$sname
mkdir $ndir
cp config.yml $ndir
mkdir $ndir"output/"
sed "s/xxx/$sname" settings.yml > $ndir"settings.yml"
#sed "s/xxx/$sname" $fdir"settings.yml" > $ndir"settings.yml"

