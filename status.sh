#!/bin/bash

HLINES=0
QRDATA=0
SCORE=0

for F in img/*png; do 
  if [ -f "${F}.hlines" ]; then
    HLINES=$(( $HLINES + 1 ))
  else
    echo "$F no hlines"  
    continue;
  fi;

  if [ -f "${F}.qrdata" ]; then
    QRDATA=$(( $QRDATA + 1 ))
  else
    echo "$F no qrdata"  
    continue;
  fi;

  if [ -f "${F}.score" ]; then
    SCORE=$(( $SCORE + 1 ))
  else
    echo "$F no score"  
    continue;
  fi;

done;

echo "total: $HLINES $QRDATA $SCORE"
