#!/bin/bash

IFS=";"
while read A B C D; do

echo "img/$C.png"

done < <( grep "img-4.*[-][0-9][0-9][0-9];;SO" list.csv ) | xargs -n1 -P1 bash -c 'G=`echo $0|sed -e "s/^img/out/" -e "s/[.]png//"`; nice -n15 timeout 60 python2.7 segment2.py $0 $G; echo $G; read T'

#  ;;SO[^;]*;[0-9];[?]

#  G=`echo $F|sed -e "s/\/[-]/\/letters\//" -e "s/[.]png//"`
#  timeout 60 python hough.py $F $G 
#done;