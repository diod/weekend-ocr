# Weekend-ocr

In fact this is OMR (Optical Mark Recognition), some pieces of OCR for
hadwritten digits left from 2017 version (hough.py) - i.e. symbol stroke
width normalization with moments etc

## Usage
1. Put scanned pdf (200 to 300dpi A4) to src/ 
(any structure, unique filenames required, for example - img-501234501.pdf
where digits are time of scan - 1st may, 23:45:01)

2. Run pdf2png-all.sh 
It unpacks pdf with *pdfimages*, single-threaded, to img/ )

3. Run runme.sh 
  * runs hough, image segmentation, qr search, checkbox detection
  * produce: img/imagename.hlines - cropbox, long line list, image rotation angle
  * img/imagename.state - final state if something wrong
  * img/imagename.qrdata - qrcode content (trim()-ed, require zbarimg)
  * img/imagename.score - 4 check-result score numbers (-1 no mark/bad detection, 0 to 7 - mark detected) 

4. Run collect.php
  * use vo2018-kirill.csv - list of mapping student reg QR code to work QR data
  * produce list.csv - list of pages with results summed-up
  * produce result.csv - aggregated list of works with status 
	(missing tasks - pages missing, no 2 checks - some task have not enough
	checks, empty - no work info found for this student, complete - data is complete)
  * pages can overwrite older ones (i.e. rescan after recheck)

5. Rerun runme.sh or reproc.sh to selectively reprocess images by grep-ing list.csv

## How it works
1. Crop image 
seeking for rows and cols with small number of active pixels

2. Make Hough transfrom (probablistic long line search), group lines by
angle, find largest group of same-angled (5deg) lines and rotate image
according to make lines horizontal (and save to cache)

3. Find QR code in the expected place 
either top right or bottom left, checking different qr-pixel size
decide to exit if we do not need searching top block here (and save qrdata to cache)
convolution-based - finds also broken QR markers, but slow

4. Find bottom and top parts of the page, by using top longest hline
and then look for 2/3 long line above, crop

5. Preprocess top-part image
Make horizontal and vertical projections to remove signatures on the right and digit
boxes below horizontal block separator. Then make projection again to find
checkbox-grid boundaries and crop again. If it was ok, then mask inter-checkbox gaps 
to separate contours.
FindContours, check if there are something like boxes 

5. Detect checkbox status
Using projections find where box is, then crop the inner part, check fill
level, use Hough to find lines and check if it looks like a cross. Outcomes
are: fill, cross, empty, unknown

6. Postprocess results
Check if we really got 4x8 grid of boxes, remove bare _something_ on fifth
line. Fix only unk in line to cross. Fix unk+fill to cross+fill. Fix
fill+fill to fill+cross.
Fail if something goes wrong.

## Dependencies
- Python 2.7, numpy, opencv 2.x
- zbar (zbarimg for QR code recognition/decoding)
- bash, pdfimages, etc for scripts/tooling

## Disclaimer
This is unoptimized, experimental code, use at your own risk :)

