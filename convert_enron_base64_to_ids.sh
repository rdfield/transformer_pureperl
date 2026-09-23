#!/usr/bin/bash

if [ ! -f "$1" ]
then
   echo "no file to process";
   exit 1
else
   LARGEFILE=$1
fi


if [[ "$2" =~ ^[0-9]+$ ]]
then
   NR=$2
else
   echo "number of split files not supplied"
   exit 1
fi
if [ ! -f "$3" ]
then
   echo "no model file supplied ";
   exit 1
else
   MODELFILE=$1
fi
if [[ -z "$4" ]]
then
   echo "file prefix not supplied";
   exit 1;
else 
   PREFIX=$4
fi

awk -v n=$NR '{print > ("${PREFIX}_" (NR-1) % n)}' $LARGEFILE

ls ${PREFIX}_* | xargs -P 6 -I {} sh -c 'perl encode_input.pl $MODELFILE {}'

cat ${PREFIX}_*.out > "${PREFIX}_$LARGEFILE.encoded"
rm ${PREFIX}_*out
rm ${PREFIX}_*
