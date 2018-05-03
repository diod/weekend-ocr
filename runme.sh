#!/bin/bash

find img/ -maxdepth 1 -name img\*.png | xargs -n1 -P3 bash -c 'G=`echo $0|sed -e "s/^img/out/" -e "s/[.]png//"`; timeout 60 python2.7 segment2.py $0 $G'

  

#  G=`echo $F|sed -e "s/\/[-]/\/letters\//" -e "s/[.]png//"`
#  timeout 60 python hough.py $F $G 
#done;