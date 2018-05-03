#!/usr/bin/python

import cv2
import numpy as np
import math
import sys
import os.path
#from matplotlib import pyplot as plt

def get_image_cropBox(bwimg, padding=2):
  rows, cols = bwimg.shape

  non_empty_columns = np.where(bwimg.max(axis=0) > 0)[0]
  non_empty_rows = np.where(bwimg.max(axis=1) > 0)[0]
  
  #x1:x2,y1:y2
  cropBox = ( min(non_empty_rows) - padding,
            min(max(non_empty_rows) + padding, rows),
            min(non_empty_columns)  - padding,
            min(max(non_empty_columns) + padding, cols))
  print "Crop box: (%d,%d) - (%d,%d)" %(cropBox[0], cropBox[2], cropBox[1], cropBox[3]);
  return cropBox

def get_page_rotate_angle(bw_img, min_line):
  img = cv2.medianBlur(bw_img, 3)

  #fix rotation
  dilate = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY)
  (_,dilate) = cv2.threshold(dilate,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  dilate = cv2.dilate(dilate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);

  #crop
  cropBox = get_image_cropBox(dilate);
  dilate =  dilate[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]

  #lines= cv2.HoughLines(edges,1,np.pi/180,200);
  lines= cv2.HoughLinesP(image=dilate,rho=1,theta=np.pi/1440,threshold=600,minLineLength=min_line,maxLineGap=5);
  print "get_angle: Hough lines: %d" % (len(lines[0]))

  avgangle = 0.0
  avgcnt = 0

  for n in range(0,len(lines)):
    for x1,y1,x2,y2 in lines[n]:
      cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

      phi = math.atan2(x1-x2, y1-y2)*180/np.pi;
      if (phi<0):
        phi+=180.0

      #horizontal lines    
      if (phi>70 and phi<110):
        avgangle+=phi-90.0
        avgcnt+=1
    
      #print "phi: %.1f (%d %d),(%d,%d)" % (phi, x1,y1, x2,y2)

  angle = -avgangle / avgcnt;
  print "get_angle: avg: cnt=%d anglediff = %0.4f" % (avgcnt, angle)

  cv2.imshow("dilate", cv2.resize(dilate, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
  cv2.waitKey(25);

  return [angle,lines,cropBox]


def im_norm_strokew_find_mu(im0, im1, tau_0, tau_1, targetT):
        tau_l = tau_0
        tau_r = tau_1
        mu_l = 0.0
        mu_r = 1.0
        
        mu = (targetT - tau_l) / (tau_l - tau_r)
        
        for xx in range(0,7):

          imT = cv2.add( cv2.multiply( im0, 1.0 - mu), cv2.multiply( im1, mu)); 
          m00 = 0.0
          for my in range(0,sym.shape[0]):
            for mx in range(0,sym.shape[1]):
              m00 += imT[my,mx]

          tauT = 1.0*m00 / (m00byrad[6]-m00byrad[5])

          print "imT[xx=%d]: mu=%0.4f m00=%0.2f tauT=%0.4f" % (xx, mu, m00, tauT)
          if (tauT > targetT): 
            tau_r = tauT
            mu_r = mu
            mu = mu_l + (mu_l - mu_r) * (targetT-tau_l) / (tau_l - tau_r)
            
          if (tauT < targetT): 
            tau_l = tauT
            mu_l = mu
            mu = mu_l + (mu_l - mu_r) * (targetT-tau_l) / (tau_l - tau_r)

        #center and size image
        m00 = 0.0
        m10 = 0.0
        m01 = 0.0
        for my in range(0,sym.shape[0]):
          for mx in range(0,sym.shape[1]):
            m00 += imT[my,mx]
            m10 += mx*imT[my,mx]
            m01 += my*imT[my,mx]

        cx = m10/m00
        cy = m01/m00

        #mu20
        mu20 = 0.0
        mu02 = 0.0
        nuh  = 0.0
        for my in range(0,sym.shape[0]):
          for mx in range(0,sym.shape[1]):
            nuh  += abs(float(mx)-cx)*imT[my,mx]
            mu20 += (float(mx)-cx)*(float(mx)-cx)*imT[my,mx]
            mu02 += (float(my)-cy)*(float(my)-cy)*imT[my,mx]

        nuh = nuh / m00
        mu20 = math.sqrt(mu20/m00)
        mu02 = math.sqrt(mu02/m00)

        print "imt: cx: %0.2f cy: %0.2f mu20: %0.2f mu02: %0.2f nuh: %0.2f" % (cx,cy, mu20, mu02, nuh)

        #keep aspect for sym image
        mu2x = max(mu20, mu02);

        return [imT, cx, cy, mu2x]


#
# Main start
#

if (len(sys.argv) < 3):
  print "Required params <src.png> <out.base>"
  sys.exit(1)

filename = sys.argv[1];
print "Reading from: %s, prefix: %s" % (filename, sys.argv[2])

#bw_img = cv2.imread('hsf_page/hsf_0/f0130_28.png') # read as it is
orig_img = cv2.imread(filename) # read as it is
im_height, im_width, im_channels = orig_img.shape

min_line = round(im_width / 3);
print "Input: %d x %d, search for lines longer than: %d" % (im_width, im_height, min_line);

#get rotation angle etc with cache
if os.path.isfile("%s.hlines" % filename):
  #read cached lines etc
  print("read hline cache from: %s.hlines" % filename)
  fd = open("%s.hlines" % filename, "r");
  angle = float(fd.readline().strip());
  cropBox = fd.readline().strip().split();
  cropBox = map(lambda x: int(x), cropBox);
  
  print cropBox;
  oldlines = [];
  
  for row in fd:
    items = row.strip().split();
    items = map(lambda x: int(x),items);
    oldlines.append(items);
  fd.close();

  oldlines = np.array([oldlines]);
  
else:
  #process image and save to cache
  (angle,oldlines, cropBox) = get_page_rotate_angle(orig_img, min_line);

  print("write hline cache to: %s.hlines" % filename)
  
  fd = open("%s.hlines" % filename, "w");
  fd.write("%0.10f\n" % angle);
  fd.write("%d %d %d %d\n" % ( cropBox[0], cropBox[1], cropBox[2], cropBox[3]) );
  for (x1,y1,x2,y2) in oldlines[0]:
    fd.write("%d %d %d %d\n" % (x1, y1, x2, y2));
  fd.close();
  

#crop
n_img   = orig_img[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
n_img = cv2.normalize(n_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX);

#rotate img
center = ( n_img.shape[1]/2, n_img.shape[0]/2 )
rot = cv2.getRotationMatrix2D(center, angle, 1.0)
rotimg = cv2.warpAffine(n_img, rot, (n_img.shape[1],n_img.shape[0]),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
#rotimg = cv2.warpAffine(n_img, rot, (n_img.shape[1],n_img.shape[0]),flags=cv2.INTER_CUBIC)

bw_img = rotimg
#rotate lines
print rot

oldlines2 = np.reshape(oldlines, (1, oldlines.shape[1]*2,2)).astype(dtype=np.float32);
newlines = cv2.transform( oldlines2, rot );
newlines = np.reshape(newlines, (1, oldlines.shape[1], 4)).astype(dtype=np.int32);

#we work now with this img:
im_height, im_width, im_channels = bw_img.shape

#bw_img = cv2.resize(bw_img, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST);
#cv2.imshow("src",bw_img);

#if bgr_img.shape[-1] == 3:           # color image
#    b,g,r = cv2.split(bgr_img)       # get b,g,r
#    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
#    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
#else:
#    gray_img = bgr_img

img = cv2.medianBlur(bw_img, 3)
#cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#dilate = cv2.cvtColor(bw_img,cv2.COLOR_BGR2GRAY)
#(_,dilate) = cv2.threshold(dilate,127,255,cv2.THRESH_BINARY_INV)
##dilate = cv2.Sobel(dilate, cv2.CV_8U, 0,1, ksize=5)
#dilate = cv2.dilate(dilate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), 1);

#dilate = cv2.erode(dilate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);

#cv2.imshow("edges", edges);
#cv2.imshow("dilate2", cv2.resize(dilate, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
#cv2.waitKey(0);

#lines= cv2.HoughLines(edges,1,np.pi/180,200);
#lines= cv2.HoughLinesP(image=dilate,rho=1,theta=np.pi/360,threshold=600,minLineLength=min_line,maxLineGap=5);
#print "Hough lines: %d (horisontal), by angle: %d" % (len(lines[0]), len(lines))

#for rho,theta in lines[0]:
#    a = np.cos(theta)
#    b = np.sin(theta)
#    x0 = a*rho
 #   y0 = b*rho
 #   x1 = int(x0 + 10000*(-b))
 #   y1 = int(y0 + 10000*(a))
 #   x2 = int(x0 - 10000*(-b))
 #   y2 = int(y0 - 10000*(a))
#
#    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

avgangle = 0.0
avgcnt = 0
#
hlines =  sorted(newlines[0], key=lambda x: ( -1*(abs(x[0]-x[2])+abs(x[1]-x[3])) ) );

linesbylen = {};

prevY = -1
prevLen = -1
prevX = -1
# group by len
for x1,y1,x2,y2 in hlines:

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    curLen = (abs(x1-x2)+abs(y1-y2));
    if (prevLen == -1):
      prevLen = curLen
    
    if (abs(prevLen-curLen)<=15): 
      #samelen
      curLen = prevLen

    #take 1 horizontal line at each Y,X
    key="%d" % (curLen)

    if key in linesbylen:
      linesbylen[key].append([x1,y1,x2,y2]);
    else:
      linesbylen[key] = [ [x1,y1,x2,y2] ];
    
    prevLen = curLen

print linesbylen

#find longest
longest_len = 0;

for key in linesbylen:
  print "len: %s, count: %d" % (key, len(linesbylen[key]))
  if len(linesbylen[key])<2: continue;
  if (int(key) > longest_len): 
    longest_len = int(key);
  
print "longest_lines: %d, count: %d" % ( longest_len, len( linesbylen[str(longest_len)] ) )

second_len = 0;
for key in linesbylen:
  if len(linesbylen[key])<3: continue;
  if (int(key) > second_len and int(key) < longest_len): 
    second_len = int(key);

if (second_len>0):
  print "second_lines: %d, count: %d" % ( second_len, len( linesbylen[str(second_len)] ) )
else:
  print "second_lines: %d, count: %d" % ( second_len, 0 )


third_len = 0;
for key in linesbylen:
  if len(linesbylen[key])<3: continue;
  if (int(key) > third_len and int(key) < second_len): 
    third_len = int(key);

if (third_len>0):
  print "third_lines: %d, count: %d" % ( third_len, len( linesbylen[str(third_len)] ) )
else:
  print "third_lines: %d, count: %d" % ( third_len, 0 )

#estimate corners
#left:
leftX = 0
rightX = 0;
botB = 0
botT = im_height 
Xc = 0;
for x1,y1,x2,y2 in linesbylen[str(longest_len)]:
  leftX += x1
  rightX += x2
  Xc+=1;
  if (botB < y1): botB = y1
  if (botT > y1): botT = y1

leftX = int(round(leftX/Xc))
rightX = int(round(rightX/Xc))

print "left: %d, right: %d, bottomTop: %d, bottomBottom: %d" % (leftX, rightX, botT, botB)

cv2.line(img,(leftX,0),(leftX,im_height-1),(255,0,0),2)
cv2.line(img,(rightX,0),(rightX,im_height-1),(255,0,0),2)

cv2.line(img,(0,botT),(im_width-1,botT),(255,0,0),2)
cv2.line(img,(0,botB),(im_width-1,botB),(255,0,0),2)

#bottom aspect
botAsp =float(botB-botT) / float(rightX-leftX);
print "* bottom block aspect: %0.4f" %(botAsp)

if (abs(botAsp - 0.4794) < 0.0050):
  print "* bottom-based detection is OK. go on";
elif (abs(botAsp - 0.4348) < 0.0050):
  print "* bottom-based detection is missing one top line, fixing";
  botT = int(botT - 0.0979*float(botB-botT))
elif (abs(botAsp - 0.3347) < 0.0050):
  print "* bottom-based detection is missing one bottom line, fixing";
  botB = int(botB + 0.1447*float(rightX-leftX))
  if (botB > im_height):
    print "** we are out of bounds, new botB: %d, height: %d" % (botB, im_height);
    cv2.imshow("hough", cv2.resize(img, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
    cv2.waitKey(0)
    sys.exit(1);
    
else:
  print "* bottom-aspect is bad, bail out";
  cv2.imshow("hough", cv2.resize(img, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
  cv2.waitKey(0)
  sys.exit(1);

cv2.line(img,(0,botT),(im_width-1,botT),(255,0,0),2)
cv2.line(img,(0,botB),(im_width-1,botB),(255,0,0),2)

cv2.imshow("hough", cv2.resize(img, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
cv2.waitKey(25);


#calc other points of image -- bottom block
w = float(rightX-leftX)
rectw = 0.0370*w
recth = 0.0410*w
rectxi = 0.1425*w
rectyi = 0.1440*w

print "b: cell: %d x %d" % (int(rectw), int(recth))

score_cells = {}

for r in range(0,3): 
  for n in range(0,6): 
    x = int(leftX + 0.0535*w + rectxi*n)
    y = int(botT+0.0650*w+rectyi*r)
    rw = int(rectw);
    rh = int(recth);
    
    c_im = img[ y-7:y+rh+12, x-7:x+rw+12 ]

    c_gray = cv2.normalize( c_im, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX);
    c_gray = cv2.cvtColor( c_gray, cv2.COLOR_BGR2GRAY);
    ret, c_thresh = cv2.threshold( c_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    ret, c_thresh = cv2.threshold( c_gray, 230, 255, cv2.THRESH_BINARY_INV); #crop threshold invert

    print "otsu thresh: %d" % (ret)

    c_thresh_copy = cv2.cvtColor(c_thresh,cv2.COLOR_GRAY2BGR);

    #get contours (of rect)    
    print c_thresh.shape
    (contours,_) = cv2.findContours(c_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, (1,1) );
    
    print "contour count: %d" % len(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True);

#    approx = cv2.approxPolyDP(contours[0], 10, True);
    approx = cv2.convexHull(cv2.approxPolyDP(contours[0], 20, True), returnPoints=True);

    if (len(approx) == 4):
      mask = np.zeros((rh+19,rw+19,3), np.float32);  #mask image
      mask[:,:] = (0,0,0)
      #draw outer rect contour with fill
      cv2.drawContours( mask, [approx], 0, (1.0,1.0,1.0), -1);
      mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), 1);

      s_im = np.zeros((rh+19,rw+19,3), np.uint8);  #mask image
      s_im[:,:] = (255,255,255);
    
      s_im = cv2.multiply(s_im, (1.0 - mask), dtype=cv2.CV_8U)
      x_im = cv2.multiply(c_im, mask, scale=1.0, dtype=cv2.CV_8U)
    
      s_im = cv2.add( s_im, x_im )

      ret, x_thresh = cv2.threshold( cv2.cvtColor(s_im, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV); #threshold invert

      (xcnt, _) = cv2.findContours(x_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,(1,1));
      
      if len(xcnt) == 0:
        score_cells[ "%d_%d" % ( r, n) ] = ' ';
        continue;
      
      #get merged bbox
      (bx,by,bw,bh) = cv2.boundingRect(xcnt[0]);
      print "xn: 0, (%d %d %d %d)" % ( bx,by,bw,bh )
      bx2 = bx+bw;
      by2 = by+bh;
      for xn in range(1, len(xcnt)):
        (bx_l,by_l,bw_l,bh_l) = cv2.boundingRect(xcnt[xn]);
        print "xn: %d, (%d %d %d %d)" % (xn, bx_l,by_l,bw_l,bh_l)
        bx2_l = bx_l + bw_l;
        by2_l = by_l + bh_l;
        if (bx_l < bx): bx = bx_l
        if (by_l < by): by = by_l
        if (bx2_l > bx2): bx2 = bx2_l
        if (by2_l > by2): by2 = by2_l
        
      bw = bx2 - bx
      bh = by2 - by

      print "bbox: (%d %d) - (%d %d), xcnt: %d, bw x bh = (%d %d)" %( bx,by,bx2,by2, len(xcnt), bw, bh)


      sym = 255 - cv2.cvtColor(s_im, cv2.COLOR_BGR2GRAY)

      m00byrad = np.zeros( 9,np.float64)
      
      for rad in range(-4,4):
        if (rad < 0):
          ksz = -2*rad+1;
          sample = cv2.erode(sym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksz,ksz)), 1);
        elif (rad>0):
          ksz = 2*rad+1;
          sample = cv2.dilate(sym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksz,ksz)), 1);
        else:
          sample = sym

        m00 = 0.0
        for my in range(0,sym.shape[0]):
          for mx in range(0,sym.shape[1]):
            m00 += sample[my,mx]

        m00byrad[rad+5] = m00
      
      tau_m3 = 2.0*m00byrad[2] / (m00byrad[3]-m00byrad[1]);  
      tau_m2 = 2.0*m00byrad[3] / (m00byrad[4]-m00byrad[2]);  
      tau_m1 = 2.0*m00byrad[4] / (m00byrad[5]-m00byrad[3]);  
      tau_0  = 2.0*m00byrad[5] / (m00byrad[6]-m00byrad[4]);  
      tau_1  = 2.0*m00byrad[6] / (m00byrad[7]-m00byrad[5]);  
      tau_2  = 2.0*m00byrad[7] / (m00byrad[8]-m00byrad[6]);
      
      print "rad:-3, tau = %2.4f" % tau_m3
      print "rad:-2, tau = %2.4f" % tau_m2
      print "rad:-1, tau = %2.4f" % tau_m1
      print "rad: 0, tau = %2.4f" % tau_0 
      print "rad: 1, tau = %2.4f" % tau_1 
      print "rad: 2, tau = %2.4f" % tau_2 

      #interpolate grey image
      targetT=1.630
      
      if (targetT > tau_0 and targetT<tau_1):
        im0 = sym
        ksz = 2*1+1;
        im1 = cv2.dilate(sym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksz,ksz)), 1);      
        tau_l = tau_0
        tau_r = tau_1

      elif (targetT > tau_m1 and targetT<tau_0):
        ksz = 2*1+1;
        im0 = cv2.erode(sym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksz,ksz)), 1);      
        im1 = sym
        tau_l = tau_m1
        tau_r = tau_0

      elif (targetT > tau_m2 and targetT<tau_m1):
        ksz = 2*1+1;
        im1 = cv2.erode(sym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksz,ksz)), 1);      
        ksz = 2*2+1;
        im0 = cv2.erode(sym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksz,ksz)), 1);      
        tau_l = tau_m2
        tau_r = tau_m1

      elif (targetT > tau_m3 and targetT<tau_m2):
        ksz = 2*2+1;
        im1 = cv2.erode(sym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksz,ksz)), 1);      
        ksz = 2*3+1;
        im0 = cv2.erode(sym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksz,ksz)), 1);      
        tau_l = tau_m3
        tau_r = tau_m2

      elif (targetT < tau_m3): #very thick
        ksz = 2*3+1;
        imT = cv2.erode(sym, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksz,ksz)), 1);
        print "non-interpolating"
        cv2.imshow("symbol", cv2.resize(c_gray, None, fx=4.0,fy=4.0, interpolation=cv2.INTER_NEAREST) );
        cv2.imshow("symOut", cv2.resize(imT, None, fx=4.0,fy=4.0, interpolation=cv2.INTER_NEAREST) );
        cv2.waitKey(0)
        continue;
        
      
      else:
        print "non-interpolating"
        cv2.imshow("symbol", cv2.resize(c_gray, None, fx=4.0,fy=4.0, interpolation=cv2.INTER_NEAREST) );
        
        c_thresh = c_gray
        
        cv2.drawContours( c_thresh, [approx], 0, (128), 1);
        cv2.imshow("symOut", cv2.resize(c_thresh, None, fx=4.0,fy=4.0, interpolation=cv2.INTER_NEAREST) );
        cv2.waitKey(0)
        continue;

      # perform stroke width norm
      (imT, cx, cy, mu2x) = im_norm_strokew_find_mu(im0, im1, tau_l, tau_r, targetT);

      imOut = np.zeros((28,28,1), np.uint8);  #grey image
      srcL = max(0, int( cx - mu2x*2.2 ))
      srcT = max(0, int( cy - mu2x*2.2 ))
      srcR = min(sym.shape[1]-1, int( cx + mu2x*2.2 ))
      srcB = min(sym.shape[0]-1, int( cy + mu2x*2.2 ))
      
      print "imt: L:%d, T: %d, R: %d, B: %d,  shape: %d %d" % (srcL, srcT, srcR, srcB, sym.shape[1], sym.shape[0]);
      
      imOut = cv2.resize(imT[srcT:srcB,srcL:srcR], (28,28), interpolation = cv2.INTER_CUBIC)
      cv2.imwrite("%s_%d_%d.png" % (sys.argv[2], r,n), imOut);
  
      cv2.imshow("symbol", cv2.resize(imT, None, fx=4.0,fy=4.0, interpolation=cv2.INTER_NEAREST) );
      cv2.imshow("symOut", cv2.resize(imOut, None, fx=4.0,fy=4.0, interpolation=cv2.INTER_NEAREST) );
      cv2.waitKey(10)


        #geometric mass center
        #cx = m10/m00
        #cy = m01/m00

#        print "rad: %d -- m00: %0.2f" % (rad, m00)
      
      
#      cv2.imshow("fuzz", cv2.resize(sample, None, fx=4.0,fy=4.0, interpolation=cv2.INTER_NEAREST) );
#      cv2.waitKey(500)
        

      

      #render
      img[ y-7:y+rh+12, x-7:x+rw+12 ] = s_im

      
    else:
      print "Bad outer contour!"
      print(approx);
      print(contours);

      cv2.imshow("symbol", cv2.resize(c_gray, None, fx=4.0,fy=4.0, interpolation=cv2.INTER_NEAREST) );

#      c_thresh = cv2.cvtColor(c_thresh_copy,cv2.COLOR_GRAY2BGR);
     
      cv2.drawContours( c_thresh_copy, [approx], 0, (0,0,255), 1);

      cv2.imshow("symOut", cv2.resize(c_thresh_copy, None, fx=4.0,fy=4.0, interpolation=cv2.INTER_NEAREST) );
      cv2.waitKey(0)
      cv2.imshow("symbol", cv2.resize(c_im, None, fx=4.0,fy=4.0, interpolation=cv2.INTER_NEAREST) );
      cv2.waitKey(0)
      sys.exit(1)
       
          
#    for c in contours:
#      approx = cv2.approxPolyDP(c, 5, True);
#      print "  area: %0.2f, vertex: %d, approx.area: %0.2f, approx.vert: %d" % (cv2.contourArea(c), len(c), cv2.contourArea(approx), len(approx) )
#      if (cv2.contourArea(c) > 3000 and len(approx) == 4):
#        cv2.drawContours( c_im, [c], -1, (255,0,0), 1);
#      else:
#        cv2.drawContours( c_im, [c], -1, (0,0,255), 1);


    
    #draw containing rect
    cv2.rectangle(img, pt1=(x-5,y-5), pt2=(x+rw+10,y+rh+10), color=(0,255,0), thickness=1);


print score_cells
sys.exit(0)


#eliminate and draw longest
prevY=-1
prevX=-1
linesbyXY = {}
for x1,y1,x2,y2 in sorted(linesbylen[str(third_len)], key=lambda x: -20000*x[1]-x[0]):

    curY = y1
    curX = x1
    if (prevY==-1):
      prevY = curY

    if (prevX==-1):
      prevX = curX

    if (abs(curY-prevY)<3):
      curY = prevY

    if (abs(curX-prevX)<3):
      curX = prevX
    
    key = "%d_%d" % (curX, curY)
    if key in linesbyXY: continue;
    
    linesbyXY[key] = [x1,y1,x2,y2]
    prevX = curX
    prevY = curY

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    if (abs(y1-y2) < 3):
      print "h: y=%d, l=%s (%d,%d) - (%d,%d)" % (y1, key, x1,y1,x2,y2)
    else:
      print "e: %d,%d - %d,%d" % (x1,y1,x2,y2)

    phi = math.atan2(x1-x2, y1-y2)*180/np.pi;
    if (phi<0):
      phi+=180.0

    #horizontal lines    
    if (phi>70 and phi<110):
      avgangle+=phi-90.0
      avgcnt+=1
    
#    print "phi: %.0f (%d %d),(%d,%d)" % (phi, x1,y1, x2,y2)

angle = -avgangle / avgcnt;
print "avg: cnt=%d anglediff = %0.4f" % (avgcnt, angle)

#rotate img
center = ( img.shape[1]/2, img.shape[0]/2 )
rot = cv2.getRotationMatrix2D(center, angle, 1.0)
rotimg = cv2.warpAffine(img, rot, (img.shape[1],img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

#make projection



#cv2.imwrite('houghlines3.jpg',img)

cv2.imshow("hough", cv2.resize(rotimg[botT:botB, leftX:rightX], None, fx=1.0,fy=1.0, interpolation=cv2.INTER_NEAREST));
cv2.waitKey(0)



#circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
#                            param1=50,param2=30,minRadius=0,maxRadius=0)

#circles = np.uint16(np.around(circles))

#for i in circles[0,:]:
#    # draw the outer circle
#    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#    # draw the center of the circle
#    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

#plt.subplot(121),plt.imshow(rgb_img)
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(cimg)
#plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
#plt.show()





