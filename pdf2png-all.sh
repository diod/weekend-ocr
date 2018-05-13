#!/bin/bash

find src/ -type f -name img-51215\*pdf | sort |while read A; do B=`basename "${A}"|sed -e s/.pdf//`; echo $A; pdfimages -png "${A}" img/$B; done;

#pdfimages -png src/img-424175521.pdf tmp/img-424175521