#!/usr/bin/python

import cv2
import numpy as np
import math
import sys
import os.path
import subprocess
import re
import collections
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
  if lines is None:
    print "get_angle: Looks like empty/invalid page -- no long lines here"
    return [ 0.0, [], cropBox ];

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

  if (avgcnt == 0):
    print "get_angle: Looks like empty/invalid page -- no long lines here"
    return [ 0.0, [], cropBox ];

  angle = -avgangle / avgcnt;
  print "get_angle: avg: cnt=%d anglediff = %0.4f" % (avgcnt, angle)

#  cv2.imshow("dilate", cv2.resize(dilate, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(25);

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


def find_qr(bw, minX, maxX, minY, maxY):
  pw = 18
  ph = 18
  # find coords which can be qr corners
  coordlist = [];
  cgroup = [];
  for x in range( minX, maxX-3*pw-1, 2):
    for y in range( minY + 3*ph, maxY , 2):
      w1  = 2*int(bw[y,x]) #black
      w1 += 0.5* (int(bw[y+ph,x]) + int(bw[y-ph,x]) + int(bw[y,x+pw]) + int(bw[y,x-pw])) #black

      w1 += 0.5 * (int(255-bw[y+2*ph, x]) + int(255-bw[y-2*ph, x]) + int(255-bw[y,x+2*pw]) + int(255-bw[y,x-2*pw])); #white
      w1 += 0.5 * (int(255-bw[y+2*ph, x+2*pw]) + int(255-bw[y-2*ph, x+2*pw]) + int(255-bw[y+2*ph,x-2*pw]) + int(255-bw[y-2*ph,x-2*pw])); #white

      w1 += int(bw[y+3*ph, x+3*pw]) + int(bw[y+3*ph, x-3*pw]) + int(bw[y-3*ph, x+3*pw]) + int(bw[y-3*ph,x-3*pw]); #black
      #white
      if (y-4*ph > 0):
        if (x+4*pw < im_width):
          w1 += 255-int(bw[y-4*ph,x+4*pw])
        else:
          w1 += 255
        w1 += 255-int(bw[y-4*ph,x-4*pw])
      else:
        w1 += 255 + 255;
      
      if (x+4*pw < im_width): 
        w1 += 255-int(bw[y+4*pw,x+4*ph]) 
      else:
        w1 += 255;
      
      w1 += 255 - int(bw[y+4*ph,x-4*pw])

      if (w1 > 3550):
#        print "%s QR anchor: %d %d" % (sys.argv[1],x,y)
        coordlist.append( (x,y) );

  #check we have 3 corners
  for pt in coordlist:
    if (len(cgroup) == 0):
      cgroup.append( [ pt ] );
      continue;

    found = 0;
    for idx,grp in enumerate(cgroup):
      g = grp[0];
    
      w = abs(g[0] - pt[0]) + abs(g[1]-pt[1]);
      if (w < 3*pw):
        found = 1;
        cgroup[idx].append(pt);
    
    if (found == 0):
      cgroup.append( [ pt ] );

  cgroup.sort(key=len,reverse=True);
#  print len(cgroup), cgroup
  
  if (len(cgroup) >=3 ): 
    for idx,grp in enumerate(cgroup):
      x0 = 0;
      y0 = 0;
      nn = 0;
      for pt in grp:
        x0 += pt[0];
        y0 += pt[1];
        nn+=1;
      cgroup[idx] = ( int(x0/nn), int(y0/nn) );
#      print cgroup
  else:
    print "QR: failed to group corners"
    print "corner-groups: ", cgroup
    return [];


  return cgroup

def get_qrdata(bw_img, filename):
  if os.path.isfile("%s.qrdata" % filename):
    print("read qrdata cache from: %s.qrdata" % filename)
    fd = open("%s.qrdata" % filename, "r");
    qrcode = fd.readline().strip();
    fd.close();
    # valid only if not empty
    if qrcode:
      return qrcode;

  #prepare image

  bw = cv2.cvtColor(bw_img, cv2.COLOR_BGR2GRAY);
  (_, bw) = cv2.threshold(bw,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);
  bw = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);

  #find qrcode probable anchors
  cgroup = find_qr(bw, int(im_width/2), im_width, 0, int(im_height/4));

  if len(cgroup) == 0: 
    return 'no-qr-found';

#  print cgroup;

  ps = 20;

  lx = cgroup[0][0];
  rx = cgroup[0][0];
  
  ty = cgroup[0][1];
  by = cgroup[0][1];
  
  for g in cgroup:
    lx = min( lx, g[0]);
    rx = max( rx, g[0]);
    ty = min( ty, g[1]);
    by = max( by, g[1]);
    
  lx = max( lx - int(3.5*ps), 0 );
  rx = rx + int(3.5*ps);

  ty = max( ty - int(3.5*ps), 0 );
  by = by + int(3.5*ps);

#  print "(%d,%d) -- (%d,%d)" %( lx, ty, rx, by )
  
  bw2 = (255 - bw[ty:by, lx:rx]);

#  cv2.imshow("hough", bw2);
#  cv2.waitKey(0)

  rows, cols = bw2.shape

  non_empty_columns = np.where(bw2.min(axis=0) < 128)[0]
  non_empty_rows = np.where(bw2.min(axis=1) < 128)[0]
  
  #x1:x2,y1:y2
  cropBox =(max(min(non_empty_rows),0),
            min(max(non_empty_rows), rows),
            max(min(non_empty_columns), 0),
            min(max(non_empty_columns), cols))
#  print "Crop box: (%d,%d) - (%d,%d)" %(cropBox[0], cropBox[2], cropBox[1], cropBox[3]);

 
#  cv2.line( img, (lx,ty), (rx,ty), (0,0,255), 2);
#  cv2.line( img, (lx,by), (rx,by), (0,0,255),2);
#  cv2.line( img, (lx,ty), (lx,by), (0,0,255),2);
#  cv2.line( img, (rx,ty), (rx,by), (0,0,255),2);


  qrfilename = "%s_qr.png" % (sys.argv[2]);
  cv2.imwrite(qrfilename, bw2[ cropBox[0]:cropBox[1]+1, cropBox[2]+1:cropBox[3]+1 ]);

  w = cropBox[3]-cropBox[2];
  h = cropBox[1]-cropBox[0]+1;

  qri = np.zeros([21,21], np.uint8);
  
  idx = 21.0
  edge = 180
  dw = w / idx;
  dh = h / idx;
  for dy in range(0,21):
    s = '';
    for dx in range(0,21):
      x = int( cropBox[2] + round( (0.5 + dx*1.0)*dw ));
      y = int( cropBox[0] + round( (0.5 + dy*1.0)*dh ));

      c = 0;
      for xx in range(x-2, x + 2):
        for yy in range(y-2, y + 2): 
          if (bw2[yy,xx] > edge): c+=1;

      if (c>12):
        s += '0';
        qri[dy,dx] = 255;
      else:
        s += '1';
        qri[dy,dx] = 0;

#      print "(%d,%d) => %d %d" %( dx,dy, bw2[y,x],c)
      
      bw2[y,x] = 127;
      
#    print(s);
    

  qrcode = subprocess.Popen(['/usr/bin/zbarimg', '-q', qrfilename], stdout=subprocess.PIPE).communicate()[0]
  qrcode = qrcode.strip().replace('QR-Code:','').strip();

  qrfilename = "%s_qr2.png" % (sys.argv[2]);
  cv2.imwrite(qrfilename, qri);

  if qrcode == '':
    #try2
    cnv = subprocess.Popen(['/usr/bin/convert', qrfilename, '-scale', '1000%', 'qr2_tmp.png' ], stdout=subprocess.PIPE).communicate()[0]
    qrcode = subprocess.Popen(['/usr/bin/zbarimg', '-q', 'qr2_tmp.png' ], stdout=subprocess.PIPE).communicate()[0]
    qrcode = qrcode.strip().replace('QR-Code:','').strip();
    
  print qrcode
  
  print("write qrdata cache to: %s.qrdata" % filename)
  
  fd = open("%s.qrdata" % filename, "w");
  fd.write("%s\n" % qrcode);
  fd.close();

#  cv2.imshow("hough", bw2[cropBox[0]:cropBox[1]+1, cropBox[2]+1:cropBox[3]+1]);
#  cv2.waitKey(0)

  return qrcode

# https://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function
def check_lcross( o1, p1, o2, p2 ):
    x = o2 - o1;
    d1 = p1 - o1;
    d2 = p2 - o2;

    cross = d1[0]*d2[1] - d1[1]*d2[0];
    if (abs(cross) < 1e-8):
      return False;

    t1 = (x[0] * d2[1] - x[1] * d2[0])/cross;
    # r = o1 + d1 * t1; -- intersection point
    if (t1 >= 0.0) and (t1 <= 1.0):
      return True;

    return False;


def check_crossing(ebox):

  gimg = cv2.cvtColor( ebox, cv2.COLOR_GRAY2BGR);

  lines=cv2.HoughLinesP(image=ebox,rho=1,theta=np.pi/360,threshold=15,minLineLength=11,maxLineGap=3);
  
  have45 = 0;
  have135 = 0;
  for (x1,y1,x2,y2) in lines[0]:
    ang = np.rad2deg( np.arctan2( (x2-x1), (y2-y1) ));
    print ang, x1,y1, x2,y2;

    #remove fantom lines from box
    if (ang > -10) and (ang < 10): continue;
    if (ang > 80) and (ang < 100): continue;
    if (ang > 175) and (ang < 185): continue;
    
    if (ang > 22) and (ang <80): have45+=1;
    if (ang > 100) and (ang < 175): have135+=1;
        
    cv2.line( gimg, (x1,y1), (x2,y2), (0,0,255), 1);

  if (have135 and have45):
    return True;

#  cv2.imshow("houghlp", cv2.resize(gimg, None, fx=7.0,fy=7.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);

#  print lines;
  return False;
  

  
def process_score(topimg):
  im_height, im_width, im_channels = topimg.shape
  print "Score_img: %d x %d (%d chan)" % (im_width, im_height, im_channels)

#  c_gray = cv2.normalize( topimg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX);
  c_gray = cv2.cvtColor( topimg, cv2.COLOR_BGR2GRAY);

#  ret, c_thresh = cv2.threshold( c_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  c_thresh = cv2.adaptiveThreshold( c_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,17,4)

  c_thresh = cv2.dilate(c_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);
  c_thresh = cv2.erode(c_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);
  
#  cv2.imshow("hough", cv2.resize(c_thresh, None, fx=2.0,fy=2.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);
  
#    ret, c_thresh = cv2.threshold( c_gray, 230, 255, cv2.THRESH_BINARY_INV); #crop threshold invert

#  print "otsu thresh: %d" % (ret)

  c_thresh_orig = c_thresh.copy();
  c_thresh_copy = cv2.cvtColor(c_thresh,cv2.COLOR_GRAY2BGR);

  #get contours (of rect)    
  print c_thresh.shape
  (contours,_) = cv2.findContours(c_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, (1,1) );
    
  print "contour count: %d" % len(contours)
  contours = sorted(contours, key = cv2.contourArea, reverse = True);

  # copy back
  c_thresh = c_thresh_orig.copy();

  # pass1: remove big rects
  for cnt in contours:
    approx = cv2.approxPolyDP(cv2.convexHull(cnt, returnPoints=True), 1.5, True);
    brect  = cv2.boundingRect(approx);
    
    if ( (brect[2] > 65) or (brect[3] > 80) ):
      cv2.fillPoly(c_thresh, [ approx ], 0);

#    if ( (brect[2] < 20) or (brect[3] < 20) ):
#      cv2.fillPoly(c_thresh, [ cnt ], 0);

#  cv2.imshow("hough", cv2.resize(c_thresh, None, fx=2.0,fy=2.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);

  # pass2
  (contours,_) = cv2.findContours(c_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, (1,1) );
    
  print "contour count: %d" % len(contours)
  contours = sorted(contours, key = cv2.contourArea, reverse = True);

  boxes = [];

  for cnt in contours:
#    approx = cv2.approxPolyDP(cnt, 10, True);
    approx = cv2.approxPolyDP(cv2.convexHull(cnt, returnPoints=True), 1.5, True);

    peri   = cv2.arcLength(approx, True)
    area   = cv2.contourArea(approx);

    brect  = cv2.boundingRect(approx);
#    print (area*2), (peri*peri/16.0), brect


    if (16*area*3.5 > peri*peri) and (len(approx)<=9) and (brect[2] > 30) and (brect[2] < 65) and (brect[3] > 30) and (brect[3] < 80):
      #check box empty
      lw = 4;
      ebox = c_thresh_orig[ brect[1]+lw:brect[1]+brect[3]-lw, brect[0]+lw:brect[0]+brect[2]-lw ];
      
      celltype = 'unk';
      nzc = cv2.countNonZero(ebox);
      tcc = (brect[3]-2*lw)*(brect[2]-2*lw);

      if nzc <= tcc*0.05:
        celltype = 'empty';
      elif nzc > tcc*0.55:
        celltype = 'fill';
      elif check_crossing(ebox):
        celltype = 'cross';  #check we have crossing
        
      print "cell (%d,%d) nzc: %d tcc: %d fill: %f type: %s" %( brect[0], brect[1], nzc, tcc, (100.0*nzc/tcc), celltype)
      
      boxes.append( [ celltype, brect ] );

      if celltype != 'unk': continue;

#      box = c_thresh_copy[ brect[1]:brect[1]+brect[3], brect[0]:brect[0]+brect[2] ];
      
#
      cv2.drawContours(c_thresh_copy, [ approx ], -1, (0,255,0),1)

      cv2.imshow("sym", cv2.resize(c_thresh_copy[ brect[1]+lw:brect[1]+brect[3]-lw, brect[0]+lw:brect[0]+brect[2]-lw ], None, fx=7.0,fy=7.0, interpolation=cv2.INTER_NEAREST));
      cv2.waitKey(0);
      

    else:
      if (len(approx)<=9) and (brect[2] > 30) and (brect[3] > 30):
        print brect, len(approx), 16*area*3.5, peri*peri
#        print approx;
        cv2.drawContours(c_thresh_copy, [ approx ], -1, (0,0,255),1)


  scfilename = "%s_top.png" % (sys.argv[2]);
  cv2.imwrite(scfilename, c_thresh_copy);


  #sort by y
  y_list = collections.OrderedDict();
  for (ct, bx) in boxes:
    y = int(bx[1]+0.5*bx[3]); #use middle y
    if (len(y_list)==0): 
      y_list[y] = [ (ct,bx) ];
      continue;
    
    found=0;
    for ly,item in y_list.iteritems():
#      print ly
      if (abs(int(ly)-y) < 20):
        found = 1;
        y_list[ly].append( (ct,bx) );
#        print 'append ',ly 

    if found==0:
      y_list[y] = [ (ct,bx) ];


  y_list = collections.OrderedDict(sorted(y_list.items()));

  #sort by x
  fail = 0;
  for key in y_list:
    x_list = y_list[key];
    x_list = sorted(x_list, key=lambda x: x[1][0]);
    y_list[key] = x_list;
    if (len(x_list) != 8): fail = 1;

  print(y_list);
    
  #proceed with validation
  if (len(y_list)!=4) or fail:
    print "Checkbox-grid is invalid (4x8 expected)";
    cv2.imshow("hough", cv2.resize(c_thresh_copy, None, fx=2.0,fy=2.0, interpolation=cv2.INTER_NEAREST));
    cv2.waitKey(0);
    return false;

  res = [];
  for key in y_list:
    rowres = -1;
    rowmisc = -1;
    x = -1;
    for (ct, bx) in y_list[key]:
      x+=1;
      if (ct == 'empty'): continue;
      if (ct == 'fill'): 
        if (rowmisc == -1):
          rowmisc = x;
          continue;
        else:
          fail = 1;
          continue;
          
      if (ct == 'unk'): 
        if (rowmisc == -1):
          rowmisc = x;
          continue;
        else:
          fail = 1;
          continue;
      if (ct == 'cross') and (rowres == -1):
        rowres = x;
        continue;
      # some kind undefined/undetected behaviour
      fail = 1;
    
    #cross miss-detected as fill
    if (rowres == -1) and (rowmisc > -1):
      rowres = rowmisc;
      
    res.append(rowres);

  print "result: ",res;
  
  if (not fail):
    return res;  
#  sorted(boxes, key=lambda bx[1][1]*1000+bx[1][0]

    
  cv2.imshow("hough", cv2.resize(c_thresh_copy, None, fx=2.0,fy=2.0, interpolation=cv2.INTER_NEAREST));
  cv2.waitKey(0);

  return [];




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

  if len(oldlines)>0:
    oldlines = np.array([oldlines]);
  else:
    oldlines = np.array([]);
  
else:
  #process image and save to cache
  (angle,oldlines, cropBox) = get_page_rotate_angle(orig_img, min_line);

  print("write hline cache to: %s.hlines" % filename)
  
  fd = open("%s.hlines" % filename, "w");
  fd.write("%0.10f\n" % angle);
  fd.write("%d %d %d %d\n" % ( cropBox[0], cropBox[1], cropBox[2], cropBox[3]) );
  if len(oldlines)>0: 
    for (x1,y1,x2,y2) in oldlines[0]:
      fd.write("%d %d %d %d\n" % (x1, y1, x2, y2));
  fd.close();

#bail out
if len(oldlines) == 0:
  fd = open("%s.state" % filename, "w");
  fd.write("empty nohlines\n");
  fd.close();
  sys.exit(0);

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

#find QR code
qrcode = get_qrdata(bw_img, filename);
if (qrcode == 'no-qr-found') or (qrcode == ''):
  print "Error: No/bad QR code found"
  fd = open("%s.state" % filename, "w");
  fd.write("no-qr-found\n");
  fd.close();
  sys.exit(1);

#R123456
if (re.match(r'^R[0-9][0-9][0-9][0-9][0-9][0-9]$', qrcode)):
  print "End for reglist: %s" % (qrcode)
  sys.exit(0);

#SO2018-4N134222P1 - no header
if (re.match(r'^SO2018[-][0-5]N[0-9][0-9][0-9][0-9][0-9][0-9]P1$', qrcode)):
  print "End for regpage - no header on p1: %s" % (qrcode)
  sys.exit(0);

#SO2018-4N201013P6 - no header
if (re.match(r'^SO2018[-][4]N[0-9][0-9][0-9][0-9][0-9][0-9]P(6|8|12)$', qrcode)):
  print "End for 4p6/6/12 - no header: %s" % (qrcode)
  sys.exit(0);

if (re.match(r'^SO2018[-][5]N[0-9][0-9][0-9][0-9][0-9][0-9]P(6|8|10|12|14)$', qrcode)):
  print "End for 4p6/6/12 - no header: %s" % (qrcode)
  sys.exit(0);


print qrcode

#cv2.imshow("hough", cv2.resize(dilate, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
#cv2.waitKey(0)

#sys.exit(0);

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
    key=curLen

    if key in linesbylen:
      linesbylen[key].append([x1,y1,x2,y2]);
    else:
      linesbylen[key] = [ [x1,y1,x2,y2] ];
    
    prevLen = curLen


linesbylen = collections.OrderedDict(sorted(linesbylen.items(), reverse=True));
print linesbylen

#find longest
longest_len = 0;

for key in linesbylen:
  print "len: %s, count: %d" % (key, len(linesbylen[key]))
  if len(linesbylen[key])<2: continue;
  if (int(key) > longest_len): 
    longest_len = int(key);
  
print "longest_lines: %d, count: %d" % ( longest_len, len( linesbylen[ longest_len ] ) )
print linesbylen[ longest_len ];


second_len = 0;
for key in linesbylen:
  if len(linesbylen[key])<3: continue;
  if (int(key) > second_len and int(key) < longest_len): 
    second_len = int(key);

if (second_len>0):
  print "second_lines: %d, count: %d" % ( second_len, len( linesbylen[ second_len ] ) )
else:
  print "second_lines: %d, count: %d" % ( second_len, 0 )


third_len = 0;
for key in linesbylen:
  if len(linesbylen[key])<3: continue;
  if (int(key) > third_len and int(key) < second_len): 
    third_len = int(key);

if (third_len>0):
  print "third_lines: %d, count: %d" % ( third_len, len( linesbylen[ third_len ] ) )
else:
  print "third_lines: %d, count: %d" % ( third_len, 0 )

#estimate corners
#left:
leftX = 0
rightX = 0;
botB = 0
botT = im_height 
Xc = 0;
# find topmost very-long (>= 95% longest) hline
for key in linesbylen:
  if key < longest_len*0.95: continue;
  for x1,y1,x2,y2 in linesbylen[key]:
    leftX += x1
    rightX += x2
    Xc+=1;
    if (botB < y1): botB = y1
    if (botT > y1): botT = y1

leftX = int(round(leftX/Xc))
rightX = int(round(rightX/Xc))

print "left: %d, right: %d, bottomTop: %d, bottomBottom: %d" % (leftX, rightX, botT, botB)


topT = 0;
topB = 0;
topL = 0;
topR = 0;

Xc = 0
Xkey = 0;
for key in linesbylen:
  for line in linesbylen[key]:
    if (line[1] < botT-10) and (key < 0.75*longest_len):
      if (Xkey == 0): Xkey = key;
      if (key != Xkey): continue;
      
      topB += line[1] + line[3];
      topL += line[0];
      topR += line[2];
      Xc+=1;
      print key,line


topT = int(round( topB / 2.0 / Xc * 0.24));
topB = int(round( topB / 2.0 / Xc * 0.95));

topL = max( int(round(topL/Xc)-2), 0);
topR = int(round(topR/Xc)+2);

print "top: (%d,%d) (%d,%d)" % (topL, topT, topR, topB)

res = process_score(bw_img[topT:topB, topL:topR]);

fd = open("%s.score" % filename, "w");
fd.write("%d %d %d %d\n" % ( res[0], res[1], res[2], res[3]) );
fd.close();


cv2.line(img,(topL-2,topT),(topL-2,topB),(255,0,0),2)
cv2.line(img,(topR+2,topT),(topR+2,topB),(255,0,0),2)

cv2.line(img,(topL,topT-2),(topR,topT-2),(255,0,0),2)
cv2.line(img,(topL,topB+2),(topR,topB+2),(255,0,0),2)


#cv2.imshow("hough", img[topT:topB, topL:topR]);
#cv2.waitKey(0);


#cv2.imshow("hough", cv2.resize(img, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
#cv2.waitKey(0)
sys.exit(0);





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
for x1,y1,x2,y2 in sorted(linesbylen[ third_len ], key=lambda x: -20000*x[1]-x[0]):

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





