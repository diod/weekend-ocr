#!/bin/bash

find img/ -maxdepth 1 -regex ".*img-51215.*[.]png" |sort| nice -n 15 xargs -n1 -P4 bash -c 'timeout 60 python2.7 segment2.py $0'

  

#  G=`echo $F|sed -e "s/\/[-]/\/letters\//" -e "s/[.]png//"`
#  timeout 60 python hough.py $F $G 
#done;
