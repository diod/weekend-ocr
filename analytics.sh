#!/bin/bash


TOTPG=0
TOTIM=0
TOTHL=0

while read A; do 
  PGC=`pdftk "${A}" dump_data|grep NumberOfPages|sed -e "s/^.*: //"`; 
  BD=`basename "${A}" | sed -e "s/[.]pdf//"`;
  
  IMC=`find img -name "$BD-*.png" -type f |wc -l`
  IMH=`find img -name "$BD-*.png.hlines" -type f |wc -l`
  
  echo $A $BD $PGC $IMC $IMH; 
  
  TOTPG=$(( $TOTPG + $PGC ))
  TOTIM=$(( $TOTIM + $IMC ))
  TOTHL=$(( $TOTHL + $IMH ))
  
done < <(find src -name img-512\*pdf -type f|sort);


echo "Total: $TOTPG $TOTIM $TOTHL"
