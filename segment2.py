#!/usr/bin/python

import cv2
import numpy as np
import math
import sys
import os.path
import subprocess
import re
import collections

import lib.findqr
#from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)



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


def find_qr(bw, minX, maxX, minY, maxY, ps):
  im_height, im_width = bw.shape
  pw = ps
  ph = ps

  # find coords which can be qr corners
  coordlist = [];
  cgroup = [];
  for x in range( max( minX, 3*pw), min( im_width - 3*pw-1, maxX), 2):
    for y in range( max( minY, 3*ph), min( maxY, im_height - 3*ph-1) , 2):
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

      if (y+4*pw < im_height):
        if (x+4*pw < im_width): 
          w1 += 255-int(bw[y+4*pw,x+4*ph]) 
        else:
          w1 += 255;
  
        w1 += 255 - int(bw[y+4*ph,x-4*pw])
      else:
          w1 += 255 + 255;
      

      if (w1 > 3550):
        print "%s %d QR anchor: %d %d  %d" % (sys.argv[1],ps,x,y,w1)
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
    nnmax = 0
    rgroup = [];
    for idx,grp in enumerate(cgroup):
      x0 = 0;
      y0 = 0;
      nn = 0;
      nnmax = max ( nnmax, len(grp) ); #sorted
      for pt in grp:
        x0 += pt[0];
        y0 += pt[1];
        nn+=1;
        
      if (nn > nnmax/2.5) and (nnmax > 7): 
        print "cgroup: (%d, %d) %d" % ( int(x0/nn), int(y0/nn), nn )
        rgroup.append( ( int(x0/nn), int(y0/nn), nn ) );

    print rgroup
    return rgroup
  else:
    print "QR: failed to group corners"
    print "corner-groups: ", cgroup
    return [];


def get_qr_cgroup(bw, im_width, im_height, tried_pxsz):
  rotate = 0;

  tests = [
    ( 0,    12, int(im_width/3),  im_width,          0,                  int(im_height/2),  0 ),

    ( 2000, 27, int(im_width/3),  im_width,          0,                  int(im_height/3),  0 ),
    ( 2000, 27, 0,                int(im_width*2/3), int(im_height*2/3), im_height,         180 ),
    ( 1700, 22, int(im_width/3),  im_width,          0,                  int(im_height/3),  0 ),
    ( 1700, 18, int(im_width/3),  im_width,          0,                  int(im_height/3),  0 ),
    ( 1340, 18, int(im_width/3),  im_width,          0,                  int(im_height/3),  0 ),
    ( 1340, 18, 0,                int(im_width*2/3), int(im_height*2/3), im_height,         180 ),
    ( 1340, 14, int(im_width/3),  im_width,          0,                  int(im_height/3),  0 ),
    ( 1340, 14, 0,                int(im_width*2/3), int(im_height*2/3), im_height,         180 ),
    ( 1150, 15, int(im_width/3),  im_width,          0,                  int(im_height/3),  0 ),
    ( 900,  13, 0,                im_width,          0,                  im_height,         0 ),
    ( 0,    9,  int(im_width/3),  im_width,          0,                  int(im_height/3),  0 ),
  ];

  print "qr: ", im_width, im_height, tried_pxsz
  
  if tried_pxsz == 0:
    tried_pxsz=100500; #any big > 27
  
  for item in tests: 
    if (im_width > item[0]) and (tried_pxsz > item[1]):
      cgroup = find_qr(bw, item[2], item[3], item[4], item[5], item[1]);
      print "cgroup@%d, rot: %d: %d" % (item[1], item[6], len(cgroup));
      if (len(cgroup)>=3):
        return (cgroup, item[6], item[1]);

    
  sys.exit(0);  
  
  
  
  
  if (im_width >=2000):
    cgroup = find_qr(bw, int(im_width/3), im_width, 0, int(im_height/3),27);
    print "cgroup@27: %d" % len(cgroup)
    if (len(cgroup)>=3):
      return (cgroup,rotate);
      
    cgroup = find_qr(bw, 0, int(im_width*2/3), int(im_height*2/3), im_height, 27);
    print "cgroup@27, rotated: %d" % len(cgroup)
    if (len(cgroup)>=3):
      rotate = 180
      return ( cgroup, rotate );

  elif (im_width >= 1700):
    cgroup = find_qr(bw, int(im_width/3), im_width, 0, int(im_height/3),22);
    print "cgroup@22: %d" % len(cgroup)
    if (len(cgroup)>=3):
      return (cgroup,rotate);
      
    cgroup = find_qr(bw, int(im_width/3), im_width, 0, int(im_height/3),18);
    print "cgroup@18: %d" % len(cgroup)
    if (len(cgroup)>=3):
      return (cgroup,rotate);
    
  elif (im_width >= 1400):
    cgroup = find_qr(bw, int(im_width/3), im_width, 0, int(im_height/3),18);
    print "cgroup@18: %d" % len(cgroup)
    if (len(cgroup)>=3) and (cgroup[0][2]>10):
      return (cgroup,rotate);
     
    cgroup = find_qr(bw, 0, int(im_width*2/3), int(im_height*2/3), im_height, 18);
    print "cgroup@18, rotated: %d" % len(cgroup)
    if (len(cgroup)>=3):
      rotate = 180
      return ( cgroup, rotate );
#    if (len(cgroup1[0]) > len(cgroup2[0])) and (len(cgroup1) >= 3):
#      return (cgroup1,rotate);
      
#    if (len(cgroup1[0]) < len(cgroup2[0])) and (len(cgroup2) >= 3):
#      rotate=180;
#      return (cgroup2,rotate);


  elif (im_width > 900):
    cgroup = find_qr(bw, int(im_width/4), im_width, 0, int(im_height/3),13);
    print "cgroup@13: %d" % len(cgroup);
    if (len(cgroup)>=3):
      return ( cgroup, rotate );

  else:
    cgroup = find_qr(bw, int(im_width/4), im_width, 0, int(im_height/3),10);
    print "cgroup@10: %d" % len(cgroup);
    if (len(cgroup)>=3):
      return ( cgroup, rotate );
  
  
  
#  cv2.imshow("hough", bw);
#  cv2.waitKey(0)
  
  if (len(cgroup)<3):
    cgroup = find_qr(bw, int(im_width/4), im_width, 0, int(im_height/3),14);
    print "cgroup@14: %d" % len(cgroup);

  if (len(cgroup)<3):
    cgroup = find_qr(bw, int(im_width/4), im_width, 0, int(im_height/3),12);
    print "cgroup@9: %d" % len(cgroup);

  if (len(cgroup)<3):
    cgroup = find_qr(bw, int(im_width/4), im_width, 0, int(im_height/3),9);
    print "cgroup@9: %d" % len(cgroup);
   

  return ( cgroup, rotate )

def binarize_qr(img, sz):
  h, w = img.shape
  print "binarizeqr: w:%d, h:%d, sz:%d" % (w,h,sz)
  qri = np.zeros([sz,sz], np.uint8);

  idx = float(sz)
  edge = 180
  dw = w / idx;
  dh = h / idx;
  for dy in range(0,sz):
    s = '';
    for dx in range(0,sz):
      x = int( round( (0.5 + dx*1.0)*dw ));
      y = int( round( (0.5 + dy*1.0)*dh ));

      c = 0;
      for xx in range(max(0,x-2), min(x+3, w-1) ): #5
        for yy in range(max(0,y-2), min(y+3, h-1) ): #4 
          if (img[yy,xx] > edge): c+=1;

      if (c>10):
        s += '0';
        qri[dy,dx] = 255;
      else:
        s += '1';
        qri[dy,dx] = 0;

#      print "(%d,%d) => %d %d" %( dx,dy, bw2[y,x],c)
#      bw2[y,x] = 127;
    print(s);

  #fix fixable, break on rotate
  if sz==21: #y,x
    qri[0,0] = 0;
    qri[1,0] = 0;
    qri[2,0] = 0;
    qri[3,0] = 0;
    qri[4,0] = 0;
    qri[5,0] = 0;
    qri[6,0] = 0;

    qri[0,20] = 0;
    qri[1,20] = 0;
    qri[2,20] = 0;
    qri[3,20] = 0;
    qri[4,20] = 0;
    qri[5,20] = 0;
    qri[6,20] = 0;
  
#    qri[17,17] = 255;
#    qri[17,18] = 255;
#    qri[17,19] = 255;

#    qri[18,17] = 255;
#    qri[18,18] = 0;
#    qri[18,19] = 255;

#    qri[19,17] = 255;
#    qri[19,18] = 255;
#    qri[19,19] = 255;

    qri[14,0] = 0;
    qri[15,0] = 0;
    qri[16,0] = 0;
    qri[17,0] = 0;
    qri[18,0] = 0;
    qri[19,0] = 0;
    qri[20,0] = 0;

    qri[14,6] = 0;
    qri[15,6] = 0;
    qri[16,6] = 0;
    qri[17,6] = 0;
    qri[18,6] = 0;
    qri[19,6] = 0;
    qri[20,6] = 0;

    qri[6,0] = 0;
    qri[6,1] = 0;
    qri[6,2] = 0;
    qri[6,3] = 0;
    qri[6,4] = 0;
    qri[6,5] = 0;
    qri[6,6] = 0;

    qri[6,14] = 0;
    qri[6,15] = 0;
    qri[6,16] = 0;
    qri[6,17] = 0;
    qri[6,18] = 0;
    qri[6,19] = 0;
    qri[6,20] = 0;

    qri[20,0] = 0;
    qri[20,1] = 0;
    qri[20,2] = 0;
    qri[20,3] = 0;
    qri[20,4] = 0;
    qri[20,5] = 0;
    qri[20,6] = 0;

  if sz==25: #y,x
    qri[0,0] = 0;
    qri[1,0] = 0;
    qri[2,0] = 0;
    qri[3,0] = 0;
    qri[4,0] = 0;
    qri[5,0] = 0;
    qri[6,0] = 0;

    qri[0,24] = 0;
    qri[1,24] = 0;
    qri[2,24] = 0;
    qri[3,24] = 0;
    qri[4,24] = 0;
    qri[5,24] = 0;
    qri[6,24] = 0;
  
    qri[17,17] = 255;
    qri[17,18] = 255;
    qri[17,19] = 255;

    qri[18,17] = 255;
    qri[18,18] = 0;
    qri[18,19] = 255;

    qri[19,17] = 255;
    qri[19,18] = 255;
    qri[19,19] = 255;

    qri[18,0] = 0;
    qri[19,0] = 0;
    qri[20,0] = 0;
    qri[21,0] = 0;
    qri[22,0] = 0;
    qri[23,0] = 0;
    qri[24,0] = 0;

  if sz==33: #y,x
    qri[0,0] = 0;
    qri[1,0] = 0;
    qri[2,0] = 0;
    qri[3,0] = 0;
    qri[4,0] = 0;
    qri[5,0] = 0;
    qri[6,0] = 0;

    qri[0,32] = 0;
    qri[1,32] = 0;
    qri[2,32] = 0;
    qri[3,32] = 0;
    qri[4,32] = 0;
    qri[5,32] = 0;
    qri[6,32] = 0;
  
#    qri[17,17] = 255;
#    qri[17,18] = 255;
#    qri[17,19] = 255;

#    qri[18,17] = 255;
#    qri[18,18] = 0;
#    qri[18,19] = 255;

#    qri[19,17] = 255;
#    qri[19,18] = 255;
#    qri[19,19] = 255;

    qri[26,0] = 0;
    qri[27,0] = 0;
    qri[28,0] = 0;
    qri[29,0] = 0;
    qri[30,0] = 0;
    qri[31,0] = 0;
    qri[32,0] = 0;


  if sz==37: #y,x
    qri[0,0] = 0;
    qri[1,0] = 0;
    qri[2,0] = 0;
    qri[3,0] = 0;
    qri[4,0] = 0;
    qri[5,0] = 0;
    qri[6,0] = 0;

    qri[0,36] = 0;
    qri[1,36] = 0;
    qri[2,36] = 0;
    qri[3,36] = 0;
    qri[4,36] = 0;
    qri[5,36] = 0;
    qri[6,36] = 0;
  
#    qri[17,17] = 255;
#    qri[17,18] = 255;
#    qri[17,19] = 255;

#    qri[18,17] = 255;
#    qri[18,18] = 0;
#    qri[18,19] = 255;

#    qri[19,17] = 255;
#    qri[19,18] = 255;
#    qri[19,19] = 255;

    qri[30,0] = 0;
    qri[31,0] = 0;
    qri[32,0] = 0;
    qri[33,0] = 0;
    qri[34,0] = 0;
    qri[35,0] = 0;
    qri[36,0] = 0;

  if sz==41: #y,x
    qri[0,0] = 0;
    qri[1,0] = 0;
    qri[2,0] = 0;
    qri[3,0] = 0;
    qri[4,0] = 0;
    qri[5,0] = 0;
    qri[6,0] = 0;

    qri[0,40] = 0;
    qri[1,40] = 0;
    qri[2,40] = 0;
    qri[3,40] = 0;
    qri[4,40] = 0;
    qri[5,40] = 0;
    qri[6,40] = 0;
  
#    qri[17,17] = 255;
#    qri[17,18] = 255;
#    qri[17,19] = 255;

#    qri[18,17] = 255;
#    qri[18,18] = 0;
#    qri[18,19] = 255;

#    qri[19,17] = 255;
#    qri[19,18] = 255;
#    qri[19,19] = 255;

    qri[34,0] = 0;
    qri[35,0] = 0;
    qri[36,0] = 0;
    qri[37,0] = 0;
    qri[38,0] = 0;
    qri[39,0] = 0;
    qri[40,0] = 0;

  return qri  

def parse_qr(bw, im_width, cgroup, tmpbase):
#  ps = im_width/90;

  lx = cgroup[0][0];
  rx = cgroup[0][0];
  
  ty = cgroup[0][1];
  by = cgroup[0][1];
  
  for g in cgroup:
    lx = min( lx, g[0]);
    rx = max( rx, g[0]);
    ty = min( ty, g[1]);
    by = max( by, g[1]);

  #max px size for 21x21 qr     
  ps = ((rx-lx)+(by-ty))/2 / 14.0

  lx = max( lx - int(3.5*ps), 0 );
  rx = rx + int(3.5*ps);

  ty = max( ty - int(3.5*ps), 0 );
  by = by + int(3.5*ps);

  print "(%d,%d) -- (%d,%d) ps:%d" %( lx, ty, rx, by, ps )
  
  bw2 = (255 - bw[ty:by, lx:rx]);

#  cv2.imshow("hough", bw2);
#  cv2.waitKey(0)

  rows, cols = bw2.shape

  proj_x = np.sum(bw2, axis=0, dtype=np.uint32);
  mpx = max(proj_x)*0.95;

  proj_y = np.sum(bw2, axis=1, dtype=np.uint32);
  mpy = max(proj_y)*0.95;
  
  print "mpx: ", mpx, " proj_x: ", proj_x  
  print "mpy: ", mpy, " proj_y: ", proj_y

  non_empty_columns = np.where(proj_x < mpx)[0]
  non_empty_rows = np.where(proj_y < mpy)[0]

  print "non_empty_cols: ", non_empty_columns
  print "non_empty_rows: ", non_empty_rows
  
  #x1:x2,y1:y2
  cropBox =(max(min(non_empty_rows),0),
            min(max(non_empty_rows), rows),
            max(min(non_empty_columns)+1, 0),
            min(max(non_empty_columns), cols))
  print "Crop box: (%d,%d) - (%d,%d)" %(cropBox[0], cropBox[2], cropBox[1], cropBox[3]);

 
  qrfilename = "%s_qr.png" % (tmpbase);
#  cv2.imwrite(qrfilename, bw2 );
  cv2.imwrite(qrfilename, bw2[ cropBox[0]:cropBox[1]+1, cropBox[2]+1:cropBox[3]+1 ]);

#  cv2.imshow("hough", bw2[ cropBox[0]:cropBox[1]+1, cropBox[2]+1:cropBox[3]+1 ]);
#  cv2.waitKey(0)

  qrcode = subprocess.Popen(['/usr/bin/zbarimg', '-q', qrfilename], stdout=subprocess.PIPE).communicate()[0]
  qrcode = qrcode.strip().replace('QR-Code:','').strip();


  if qrcode == '':
    #try2
    qri = binarize_qr(bw2[ cropBox[0]:cropBox[1]+1, cropBox[2]+1:cropBox[3]+1 ], 21);
#    qri = binarize_qr(bw2, 21);

    qrfilename = "%s_qr21.png" % (tmpbase);
    qrb_file   = "%s_qr21b.png" % (tmpbase);
    cv2.imwrite(qrfilename, qri);
    
    cnv = subprocess.Popen(['/usr/bin/convert', qrfilename, '-scale', '1000%', qrb_file ], stdout=subprocess.PIPE).communicate()[0]
    qrcode = subprocess.Popen(['/usr/bin/zbarimg', '-q', qrb_file ], stdout=subprocess.PIPE).communicate()[0]
    qrcode = qrcode.strip().replace('QR-Code:','').strip();

  if qrcode == '':
    #try3
    qri = binarize_qr(bw2[ cropBox[0]:cropBox[1]+1, cropBox[2]+1:cropBox[3]+1 ], 33);
#    qri = binarize_qr(bw2, 33);

    qrfilename = "%s_qr33.png" % (tmpbase);
    qrb_file   = "%s_qr33b.png" % (tmpbase);
    cv2.imwrite(qrfilename, qri);
    
    cnv = subprocess.Popen(['/usr/bin/convert', qrfilename, '-scale', '1000%', qrb_file ], stdout=subprocess.PIPE).communicate()[0]
    qrcode = subprocess.Popen(['/usr/bin/zbarimg', '-q', qrb_file ], stdout=subprocess.PIPE).communicate()[0]
    qrcode = qrcode.strip().replace('QR-Code:','').strip();

  if qrcode == '':
    #try4
    qri = binarize_qr(bw2[ cropBox[0]:cropBox[1]+1, cropBox[2]+1:cropBox[3]+1 ], 37);
#    qri = binarize_qr(bw2, 37);

    qrfilename = "%s_qr37.png" % (tmpbase);
    qrb_file   = "%s_qr37b.png" % (tmpbase);
    cv2.imwrite(qrfilename, qri);
    
    cnv = subprocess.Popen(['/usr/bin/convert', qrfilename, '-scale', '1000%', qrb_file ], stdout=subprocess.PIPE).communicate()[0]
    qrcode = subprocess.Popen(['/usr/bin/zbarimg', '-q', qrb_file ], stdout=subprocess.PIPE).communicate()[0]
    qrcode = qrcode.strip().replace('QR-Code:','').strip();

  if qrcode == '':
    #try4
    qri = binarize_qr(bw2[ cropBox[0]:cropBox[1]+1, cropBox[2]+1:cropBox[3]+1 ], 41);
#    qri = binarize_qr(bw2, 41);

    qrfilename = "%s_qr41.png" % (tmpbase);
    qrb_file   = "%s_qr41b.png" % (tmpbase);
    cv2.imwrite(qrfilename, qri);
    
    cnv = subprocess.Popen(['/usr/bin/convert', qrfilename, '-scale', '1000%', qrb_file ], stdout=subprocess.PIPE).communicate()[0]
    qrcode = subprocess.Popen(['/usr/bin/zbarimg', '-q', qrb_file ], stdout=subprocess.PIPE).communicate()[0]
    qrcode = qrcode.strip().replace('QR-Code:','').strip();
    
  print qrcode
  return qrcode


def get_qrdata(bw_img, filename):
  if os.path.isfile("%s.qrdata" % filename):
    print("read qrdata cache from: %s.qrdata" % filename)
    fd = open("%s.qrdata" % filename, "r");
    qrdata = fd.readline().strip().split(' ');
    fd.close();

#    print qrdata, len(qrdata), len(qrdata[0])

    # valid only if not empty, support no rotate data
    if (len(qrdata) == 2) and ( len(qrdata[0]) >=7):
      return (qrdata[0], int(qrdata[1]))
      
    if (len(qrdata) == 1) and ( len(qrdata[0]) >=7):
      return (qrdata[0], 0 )

  #prepare image

  bw = cv2.cvtColor(bw_img, cv2.COLOR_BGR2GRAY);
  (_, bw) = cv2.threshold(bw,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

  bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);
  bw = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);

#  cv2.imshow("hough", bw);
#  cv2.waitKey(0)

  #find qrcode probable anchors
  im_height, im_width = bw.shape
  tried_pxsz = 0;
  
  (cgroup, rotate, tried_pxsz) = get_qr_cgroup(bw, im_width, im_height, tried_pxsz);

  if len(cgroup) == 0: 
    return ('no-qr-found',0);

  qrcode = parse_qr(bw, im_width, cgroup, tmpbase);
  if len(qrcode)==0:
    (cgroup, rotate, tried_pxsz) = get_qr_cgroup(bw, im_width, im_height, tried_pxsz);
    if len(cgroup) == 0: 
      return ('no-qr-found',0);

    qrcode = parse_qr(bw, im_width, cgroup, tmpbase);
      
  if len(qrcode)==0:
    (cgroup, rotate, tried_pxsz) = get_qr_cgroup(bw, im_width, im_height, tried_pxsz);
    if len(cgroup) == 0: 
      return ('no-qr-found',0);

    qrcode = parse_qr(bw, im_width, cgroup, tmpbase);


#  print cgroup;

  
  print("write qrdata cache to: %s.qrdata" % filename)
  
  fd = open("%s.qrdata" % filename, "w");
  fd.write("%s %d\n" % (qrcode, rotate));
  fd.close();

#  cv2.imshow("hough", bw2[cropBox[0]:cropBox[1]+1, cropBox[2]+1:cropBox[3]+1]);
#  cv2.waitKey(0)

  return (qrcode, rotate)

# https://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function
def check_lcross( ln1, ln2 ):
    o1x = ln1[0];
    o1y = ln1[1];
    p1x = ln1[2];
    p1y = ln1[3];

    o2x = ln2[0];
    o2y = ln2[1];
    p2x = ln2[2];
    p2y = ln2[3];
    
    xx = o2x - o1x;
    xy = o2y - o1y;
    
    d1x = p1x - o1x
    d1y = p1y - o1y
    d2x = p2x - o2x
    d2y = p2y - o2y
    
#    x = o2 - o1;
#    d1 = p1 - o1;
#    d2 = p2 - o2;

#    cross = d1[0]*d2[1] - d1[1]*d2[0];
    cross = d1y*d2x - d1x*d2y;
    if (abs(cross) < 1e-8):
      return False;

    t1 = (xy * d2x - xx * d2y)/cross;
    t2 = (xy * d1x - xx * d1y)/cross;
    # r = o1 + d1 * t1; -- intersection point
    if (t1 >= 0.0) and (t1 <= 1.0) and (t2 >= 0.0) and (t2 <= 1.0):
      return True;


    return False;


def check_crossing(ebox):
  imh, imw = ebox.shape;

  #x-axis projection
  proj_x = np.sum(ebox, axis=0, dtype=np.uint32);
  mpx = max(proj_x)*0.7;
  
  xlist = np.where( proj_x > mpx )[0]

  if (len(xlist) == 0):
    print " empty: xlist ", xlist
    return ('empty',0.0);
  
#  if (len(xlist)<=2):
#    print " empty: xlist ", xlist
#    return ('empty',0.0);
    
  Lminx = min(xlist);
  Rmaxx = max(xlist);

  if (Lminx > int(0.3*imw)):
    Lminx = 0;
  if (Rmaxx < int(0.7*imw)):
    Rmaxx = imw-1;
 
  x0 = Lminx
  Lmaxx = x0
  flag=0
  for x in xlist:
    if (x-x0>1):
      Lmaxx = x0;
      flag=1
      break;
    x0 = x  
  if flag==0:
    Lmaxx = x0
      
  x0 = Rmaxx
  Rminx = x0
  flag=0
  for x in reversed(xlist):
    if (x0-x>1):
      Rminx = x0;
      flag=1
      break;
    x0 = x  
  if flag==0:
    Rminx = x0

  #dont remove too much
  if (Lmaxx-Lminx > 7):
    Lmaxx = Lminx+7;
  
  if (Rmaxx-Rminx > 7):
    Rminx = Rmaxx-7
      
#  print "xlist: ", xlist
  print "xboundaries: ", (Lminx, Lmaxx, Rminx, Rmaxx);

  proj_y = np.sum(ebox, axis=1, dtype=np.uint32);
  mpy = max(proj_y)*0.7;
  
  ylist = np.where( proj_y > mpy )[0]
  if (len(ylist) == 0):
    print " empty: ylist ", ylist
    return ('empty',0);
    
#  if (len(ylist)<2):
#    print " empty: ylist ", ylist
#    return 'unk';
  
  Tminy = min(ylist);
  Bmaxy = max(ylist);
 
  y0 = Tminy
  Tmaxy = y0
  flag = 0
  for y in ylist:
    if (y-y0>1):
      Tmaxy = y0;
      flag = 1
      break;
    y0 = y
  if flag==0:
    Tmaxy = y0
 
  y0 = Bmaxy
  Bminy = y0
  flag = 0
  for y in reversed(ylist):
    if (y0-y>1):
      Bminy = y0;
      flag = 1
      break;
    y0 = y  
  if flag==0:
    Bminy = y0;
  
  #dont remove too much
  if (Tmaxy-Tminy > 7):
    Tmaxy = Tminy+7;
  
  if (Bmaxy-Bminy > 7):
    Bminy = Bmaxy-7

#  print "ylist: ", ylist
  print "yboundaries: ", (Tminy, Tmaxy, Bminy, Bmaxy);

  cropT = Tmaxy+2;
  cropB = Bminy-2;
  cropL = Lmaxx+3;
  cropR = Rminx-2;
  
  if (Lmaxx > Rminx): #1vert line
    if (Lmaxx > 0.5*imw):
      cropL = 0+2;
      cropR = Rminx-2;
    else:
      cropL = Lmaxx+3;
      cropR = imw-2;

  if (Tmaxy > Bminy): #1horiz line
    if (Tmaxy > 0.5*imh): #bottom line
      cropT = 0+2;
      cropB = Bminy-2;
    else:
      cropT = Tmaxy+2;
      cropB = imh-2;  
  
  #cropbox
  xebox = ebox[ cropT:cropB, cropL:cropR ];  

  imh, imw = xebox.shape;

  #check pixel count
  nzc = cv2.countNonZero(xebox);
  tcc = imh*imw;

  if (tcc == 0):
    print " inner: nzc=%d, tcc=%d" %( nzc, tcc);
    return ('empty', 0.0);
  
  #fill
  fill = 100.0*nzc/tcc;

  print " inner: nzc=%d, tcc=%d, fill=%f" %( nzc, tcc, fill);

  if (nzc <= tcc*0.020):
    return ('empty', fill);

  if (nzc >= tcc*0.55):
    return ('fill', fill);

#  cv2.imshow("houghlp", cv2.resize(xebox, None, fx=7.0,fy=7.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);

  gimg = cv2.cvtColor( xebox, cv2.COLOR_GRAY2BGR);

  # find lines
#  lines=cv2.HoughLinesP(image=xebox,rho=1,theta=np.pi/360,threshold=7,minLineLength=8,maxLineGap=3);
  lines=cv2.HoughLinesP(image=xebox,rho=1,theta=np.pi/360,threshold=10,minLineLength=8,maxLineGap=4);

  if lines is None:
    print " xlines: 0" 
    if (nzc >= tcc*0.398):
      return ('fill',fill);
    return ('empty', fill);

  print " xlines: %d" % len(lines[0]);

  have45 = 0;
  have135 = 0;
  
  lines0 = [];
  lines45 = [];
  lines90 = [];
  lines135 = [];

  crosses = 0;
  checks = 0;
  for i in range (0, len(lines[0])):
    ln1 = lines[0][i]
    have_cross = 0;
    for j in range( i+1, len(lines[0])):
      ln2 = lines[0][j];
      checks = checks+1;
      if check_lcross( ln1, ln2 ):
        crosses = crosses+1;
        have_cross = 1;

    if (have_cross == 1):
      cv2.line( gimg, (ln1[0],ln1[1]), (ln1[2],ln1[3]), (0,255,0), 1);
    else:
      cv2.line( gimg, (ln1[0],ln1[1]), (ln1[2],ln1[3]), (255,0,0), 1);

  print " checks: %d, crosses: %d" % (checks,crosses);

  if (checks == 0):
    return ('empty', fill);

  if (len(lines[0])>=12) and (fill >= 0.35):
    return ('fill', fill);

  if crosses * 2.7 >= checks:
    return ('cross', fill);

  if (nzc >= tcc*0.398):
    return ('fill', fill);

#  cv2.imshow("houghlp", cv2.resize(gimg, None, fx=7.0,fy=7.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);

#    ang = np.rad2deg( np.arctan2( (x2-x1), (y2-y1) ));
#    print ang, x1,y1, x2,y2;

    #remove fantom lines from box
#    if (ang > -10) and (ang < 10): #vert
#      x = (x1+x2)/2.0;
#      if (Lminx <= x) and (x >= Lmaxx): #remove
#        continue;
#      if (Rminx <= x) and (x >= Rmaxx): #remove
#        continue;
#      lines0.append( (x1,y1,x2,y2) );
#      continue;
#    if (ang > 80) and (ang < 100):
#      lines90.append( (x1,y1,x2,y2) );
#      continue;
#    if (ang > 175) and (ang < 185): 
#      x = (x1+x2)/2.0;
#      if (Lminx <= x) and (x >= Lmaxx): #remove
#        continue;
#      if (Rminx <= x) and (x >= Rmaxx): #remove
#        continue;
#      lines0.append( (x1,y1,x2,y2) );
#      continue;


#    cv2.line( gimg, (x1,y1), (x2,y2), (0,0,255), 1);

    
#    if (ang > 10) and (ang <79): 
#      have45+=1;
#      lines45.append( (x1,y1,x2,y2) );
#    if (ang > 103) and (ang < 175): 
#      have135+=1;
#      lines135.append( (x1,y1,x2,y2) );

#  for (x1,y1,x2,y2) in lines0:
#    x = (x1+x2)/2;
#    if (LminX <= x) and (x >= LmaxX):
#      cv2.line( gimg, (x1,y1), (x2,y2), (255,0,0), 1);
#    if (RminX <= x) and (x >= RmaxX):
#      cv2.line( gimg, (x1,y1), (x2,y2), (255,0,0), 1);
    


  
  print lines0


#  cv2.imshow("houghlp", cv2.resize(gimg, None, fx=7.0,fy=7.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);


#  if (have135 and have45):
#    return 'cross';
#
#  if not have135 and not have45:
#    return 'empty';
    



#  print lines;
  return ('unk',fill);

  

  
def process_score(topimg, topw, grid_items_x):
  im_height, im_width, im_channels = topimg.shape
  print "Score_img: %d x %d (%d chan), grid-items-x: %d" % (im_width, im_height, im_channels, grid_items_x)

  grid_items_y = 4

#  c_gray = cv2.normalize( topimg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX);
  c_gray = cv2.cvtColor( topimg, cv2.COLOR_BGR2GRAY);

#  ret, c_thresh = cv2.threshold( c_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  c_thresh = cv2.adaptiveThreshold( c_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,71,43)

  c_thresh = cv2.dilate(c_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);
  c_thresh = cv2.erode(c_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);
 
#  cv2.imshow("hough", cv2.resize(c_thresh, None, fx=1.2,fy=1.2, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);
  
#    ret, c_thresh = cv2.threshold( c_gray, 230, 255, cv2.THRESH_BINARY_INV); #crop threshold invert

#  print "otsu thresh: %d" % (ret)

  # pass1: try h/v projection for crop
  proj_x = np.sum(c_thresh, axis=0, dtype=np.uint32);
  top_mpx = max(proj_x)
  print "top_mpx: ", top_mpx, " proj_x: ", proj_x

  # mediana - style 
  sorted_pjx = np.sort(proj_x);
  print "spx: ", sorted_pjx
  
  if len(sorted_pjx) > 30:
    med_mpx = sorted_pjx[ len(sorted_pjx) - 15 ];

  mpx = med_mpx*0.75;
    
  xlist = np.where( proj_x > mpx )[0]
  print "p1: med_mpx: ", med_mpx, " mpx: ", mpx, ", xlist: ", mpx/255, xlist
  

  proj_y = np.sum(c_thresh, axis=1, dtype=np.uint32);
  mpy = max(proj_y)*0.6;
  print "mpy: ", mpy, " proj_y: ", proj_y
  
  #mediana
  sorted_pjy = np.sort(proj_y);
  print "spy: ", sorted_pjy
    
  if len(sorted_pjy) > 300:
    med_mpy = sorted_pjy[ len(sorted_pjy) - 25 ];
  else:
    med_mpy = sorted_pjy[ len(sorted_pjy) - 15 ];
 
  mpy = med_mpy*0.95;  

  #remove peaks - questionable
  for y in range(1,len(proj_y)-2):
    if (proj_y[y-1] < mpy) and (proj_y[y] > mpy) and (proj_y[y+1] < mpy):
      proj_y[y] = proj_y[y-1];
    if (proj_y[y] > mpy*99.0):
      proj_y[y] = 0;
      proj_y[y-1] = 0;
      proj_y[y-2] = 0;
      proj_y[y+1] = 0;
      proj_y[y+2] = 0;

  if (proj_y[len(proj_y)-1] > mpy*1.5):
    proj_y[len(proj_y)-1] = 0;

  print "mpy: ", mpy, " proj_y: ", proj_y

  
  ylist = np.where( proj_y >= mpy)[0]
  print "p1: med_mpy: ", med_mpy, " mpy: ", mpy, ", ylist: ", mpy/255, ylist
  

  maxY = max(ylist); #-int(im_height/38);
  maxX = max(xlist);
  minY = max(0, min(ylist)-1);
  minX = max(0, min(xlist)-1);

  if (top_mpx > med_mpx*1.20): 
    maxX = maxX-3;

  #do not crop too much
  if (maxX < im_width*0.71):
    maxX = im_width
  if (maxY < im_height*0.75):
    maxY = im_height

  print "p1: crop %dx%d to: (%d:%d)x(%d:%d) " % (im_width,im_height, minX, maxX, minY, maxY)
  c_thresh = c_thresh[ minY:maxY, minX:maxX ]  

#  cv2.imshow("hough", cv2.resize(c_thresh, None, fx=1.2,fy=1.2, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);

  im_height = maxY-minY
  im_width = maxX-minX

  #pass2: get grid maybe
  nomask = 0;
  
  proj_x = np.sum(c_thresh, axis=0, dtype=np.uint32);
  mpx = max(proj_x)*0.8;
  
  xlist = np.where( proj_x > mpx )[0]
  print "p2: xlist: ", mpx/255, xlist
  
  proj_y = np.sum(c_thresh, axis=1, dtype=np.uint32);
#  mpy = max(proj_y)*0.7;
  mpy = min(im_width*255*0.40, max(proj_y)*0.7);
  
  ylist = np.where( proj_y > mpy )[0]
  print "p2: ylist: ", mpy/255, ylist
  
  minY = max(min(ylist)-1, 0);
  maxY = min(max(ylist)+1, im_height);
  
  minX = max(min(xlist)-1, 0);
  maxX = min(max(xlist)+1, im_width);

  if minX>0.2*im_width:
    minX = 0;
    nomask = 1;
  if maxX < 0.8*im_width:
    maxX = im_width-1;
    nomask = 1;
  if minY>0.2*im_height:
    minY = 0;
    nomask = 1;
  if maxY < 0.8*im_height:
    maxY = im_height-1;
    nomask = 1;

#  cv2.imshow("hough", cv2.resize(c_thresh[minY:maxY, minX:maxX], None, fx=2.0,fy=2.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);

  #do not crop too much
  if (maxX < im_width*0.8):
    maxX = im_width
  if (maxY < im_height*0.8):
    maxY = im_height

  c_thresh = c_thresh[ minY:maxY, minX:maxX ]  
  im_width = maxX-minX+1
  im_height = maxY-minY+1

  print "p2: crop to: ", im_width,im_height


  #mask
  if nomask==0:
    w = (im_width-3)/float(grid_items_x*2-1.0);
    for n in range (1,grid_items_x*2,2):
      x0 = int( w*n )+6;
      x1 = int( w*(n+1))-3;
      for x in range(x0,x1):
        cv2.line(c_thresh,(x,0),(x,im_height-1),(0,0,0),2)

    h = (im_height-3)/float(grid_items_y*2-1.0);
    for n in range (1,grid_items_y*2,2):
      y0 = int( h*n )+6;
      y1 = int( h*(n+1))-3;
      for y in range(y0,y1):
        cv2.line(c_thresh,(0,y),(im_width-1, y),(0,0,0),2)


  c_thresh_orig = c_thresh.copy();
  c_thresh_copy = cv2.cvtColor(c_thresh,cv2.COLOR_GRAY2BGR);



#  cv2.imshow("hough", cv2.resize(c_thresh_copy, None, fx=2.0,fy=2.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);


  #get contours (of rect)    
  (contours,_) = cv2.findContours(c_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, (1,1) );
    
  print "contour count: %d" % len(contours)
  contours = sorted(contours, key = cv2.contourArea, reverse = True);

  # copy back
  c_thresh = c_thresh_orig.copy();

  w30 = int(topw/31.6)
  w28 = int(topw/33.9)-1
  w65 = int(topw/14.61)
  w80 = int(topw/11.9)
  w400= int(topw/4)

  # pass1: remove big rects
  for cnt in contours:
    approx = cv2.approxPolyDP(cv2.convexHull(cnt, returnPoints=True), 2, True);
    brect  = cv2.boundingRect(approx);
    
    if ( (brect[2] > w65) or (brect[3] > w80) ) and (brect[3] < w400):
      cv2.fillPoly(c_thresh, [ approx ], 0);

#    if ( (brect[2] < 20) or (brect[3] < 20) ):
#      cv2.fillPoly(c_thresh, [ cnt ], 0);

#  cv2.imshow("hough", cv2.resize(c_thresh, None, fx=1.0,fy=1.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);

  # pass2
  (contours,_) = cv2.findContours(c_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, (1,1) );
    
  print "contour count(2): %d" % len(contours)
  contours = sorted(contours, key = cv2.contourArea, reverse = True);

  boxes = [];


  for cnt in contours:
#    approx = cv2.approxPolyDP(cnt, 10, True);
    approx = cv2.approxPolyDP(cv2.convexHull(cnt, returnPoints=True), 1.5, True);

    peri   = cv2.arcLength(approx, True)
    area   = cv2.contourArea(approx);

    brect  = cv2.boundingRect(approx);
#    print "cnt: ", (area*3.5), (peri*peri/16.0), brect, w28, w65


    if (16*area*3.5 > peri*peri) and (len(approx)<=9) and (brect[2] >= w28) and (brect[2] <= w65) and (brect[3] >= w28) and (brect[3] <= w80):
      #check box empty
      lw = 4;

      (celltype, fill) = check_crossing(c_thresh_orig[ brect[1]:brect[1]+brect[3], brect[0]:brect[0]+brect[2] ]);
      
      ebox = c_thresh_orig[ brect[1]+lw:brect[1]+brect[3]-lw, brect[0]+lw:brect[0]+brect[2]-lw ];
      
      print "cell (%d,%d) fill: %f type: %s" %( brect[0], brect[1], fill, celltype)
      
      boxes.append( [ celltype, fill, brect ] );

      if celltype != 'unk': continue;

#      box = c_thresh_copy[ brect[1]:brect[1]+brect[3], brect[0]:brect[0]+brect[2] ];
      
#
      cv2.drawContours(c_thresh_copy, [ approx ], -1, (0,255,0),1)


#      cv2.imshow("sym", cv2.resize(c_thresh_copy[ brect[1]+lw:brect[1]+brect[3]-lw, brect[0]+lw:brect[0]+brect[2]-lw ], None, fx=2.0,fy=2.0, interpolation=cv2.INTER_NEAREST));
#      cv2.waitKey(0);

      
    else:
      if (len(approx)<=9) and (brect[2] > w28) and (brect[3] > w28):
        print brect, len(approx), 16*area*3.5, peri*peri
#        print approx;
        cv2.drawContours(c_thresh_copy, [ approx ], -1, (0,0,255),1)


#  cv2.imshow("hough", cv2.resize(c_thresh_copy, None, fx=1.0,fy=1.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);

  scfilename = "%s_top.png" % (tmpbase);
  cv2.imwrite(scfilename, c_thresh_copy);


  #sort by y
  y_list = collections.OrderedDict();
  for (ct, fi, bx) in boxes:
    y = int(bx[1]+0.5*bx[3]); #use middle y
    if (len(y_list)==0): 
      y_list[y] = [ (ct,fi,bx) ];
      continue;
    
    found=0;
    for ly,item in y_list.iteritems():
#      print ly
      if (abs(int(ly)-y) < 20):
        found = 1;
        y_list[ly].append( (ct,fi,bx) );
#        print 'append ',ly 

    if found==0:
      y_list[y] = [ (ct,fi,bx) ];


  y_list = collections.OrderedDict(sorted(y_list.items()));

  #sort by x
  fail = 0;
  for key in y_list:
    x_list = y_list[key];
    x_list = sorted(x_list, key=lambda x: x[2][0]);
    y_list[key] = x_list;
    if (len(x_list) == 1): 
      #odd item
      print "remove bare item line: %d" % key, x_list;
      y_list.pop(key,None);
      continue;
    
    if (len(x_list) != grid_items_x): fail = 1;


  print(y_list);
    
  #proceed with validation
  if (len(y_list)!=grid_items_y) or fail:
    print "Checkbox-grid is invalid (%dx%d expected)" %(grid_items_y, grid_items_x);
#    cv2.imshow("hough", cv2.resize(c_thresh_copy, None, fx=1.5,fy=1.5, interpolation=cv2.INTER_NEAREST));
#    cv2.waitKey(0);
    return False;

  #check for double fill
  for key in y_list:
    cross_l = [];
    fill_l = [];
    empty_l = [];
    unk_l = [];
    x = -1;
    for (ct, fi, bx) in y_list[key]:
      x+=1;
      if (ct == 'fill'): 
        fill_l.append(x);
      if (ct == 'cross'): 
        cross_l.append(x)
      if (ct == 'empty'): 
        empty_l.append(x)
      if (ct == 'unk'): 
        unk_l.append(x)
        
    if (len(cross_l) == 1): continue; #no fixing
    if (len(cross_l) == 0) and (len(fill_l) == 2) and (len(unk_l)==0):
      print " autofix double fill: ",key, fill_l;
      print y_list[key];
      fi1 = y_list[key][fill_l[0]][1];
      fi2 = y_list[key][fill_l[1]][1];
      if (fi1 > fi2):
        y_list[key][fill_l[1]] = ('cross', fi2, y_list[key][fill_l[1]][2]);
      else:
        y_list[key][fill_l[0]] = ('cross', fi1, y_list[key][fill_l[0]][2]);
      continue;

    if (len(cross_l) == 0) and (len(fill_l) == 1) and (len(unk_l)==1):
      print " autofix unk+fill: ", key, unk_l, fill_l
      unk_fi = y_list[key][unk_l[0]][1];
      if (unk_fi >10 and unk_fi < 40):
        y_list[key][unk_l[0]] = ('cross', unk_fi, y_list[key][unk_l[0]][2]);
      continue;

    if (len(cross_l) == 0) and (len(fill_l) == 0) and (len(unk_l) == 0):
      #all empty, check if we mis-detected cross
      prob_cross = [];
      x = -1;
      for (ct,fi,bx) in y_list[key]:
        x = x + 1;
        if (ct == 'empty') and (fi > 6.0):
          prob_cross.append(x)
      if (len(prob_cross) == 1):
        print " autofix empty line with cross @", prob_cross[0];
        fi = y_list[key][prob_cross[0]][1];
        bx = y_list[key][prob_cross[0]][2];
        y_list[key][prob_cross[0]] = ('cross', fi, bx );

  res = [];
  for key in y_list:
    rowres = -1;
    rowmisc_l = [];
    x = -1;
    for (ct, fi, bx) in y_list[key]:
      x+=1;
      if (ct == 'empty'): continue;
      if (ct == 'fill'): 
        rowmisc_l.append(x);
        continue;
          
      if (ct == 'unk'): 
        rowmisc_l.append(x);
        continue;

      if (ct == 'cross') and (rowres == -1):
        rowres = x;
        continue;
      # some kind undefined/undetected behaviour
      fail = 1;
    
    #cross miss-detected as fill or unk
    if (rowres == -1) and (len(rowmisc_l) == 1):
      rowres = rowmisc_l[0];
      
    res.append(rowres);

  print "fail: %d, result: " % fail, res;
#  cv2.imshow("hough", cv2.resize(c_thresh_copy, None, fx=2.0,fy=2.0, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);
  
  if (not fail):
    return res;  
#  sorted(boxes, key=lambda bx[1][1]*1000+bx[1][0]

    

  return [];




#
# Main start
#

if (len(sys.argv) < 2):
  print "Required params <src.png>"
  sys.exit(1)

filename = sys.argv[1];
tmpbase = re.sub(r'img/', 'out/', filename);
tmpbase = re.sub(r'[.]png$', '', tmpbase);

print "Reading from: %s, prefix: %s" % (filename, tmpbase)

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
n_img = orig_img[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]
n_img = cv2.normalize(n_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX);

#check rotate
if (im_width>im_height) and (angle > -10) and (angle < 10):
  angle -= 90

#rotate
angle_r = angle * np.pi/180.0;
_sin = abs(math.sin(angle_r));
_cos = abs(math.cos(angle_r));

(h,w,chan) = n_img.shape;

newWidth  = int(round(w * _cos + h * _sin));
newHeight = int(round(w * _sin + h * _cos));

print "nW: %d, nH: %d" % (newWidth, newHeight)



#cv2.imshow("hough", cv2.resize(n_img, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
#cv2.waitKey(0)

#rotate img
center = ( int(h/2.0), int(h/2.0) )
rot = cv2.getRotationMatrix2D(center, angle, 1.0)
rotimg = cv2.warpAffine(n_img, rot, (newWidth,newHeight),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

#rotimg = cv2.warpAffine(n_img, rot, (n_img.shape[1],n_img.shape[0]),flags=cv2.INTER_CUBIC)

(h,w,chan) = n_img.shape;
print "h,w,angle: %d,%d,%f" %(h,w,angle)
print "center: ", center

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
(qrcode, rotate) = get_qrdata(bw_img, filename);
if (qrcode == 'no-qr-found') or (qrcode == ''):
  print "Error: No/bad QR code found"
  fd = open("%s.state" % filename, "w");
  fd.write("no-qr-found\n");
  fd.close();
  sys.exit(1);

#R123456
if (re.match(r'^R[0-9][0-9][0-9][0-9][0-9][0-9][0-9]$', qrcode)):
  print "End for reglist: %s" % (qrcode)
  fd = open("%s.state" % filename, "w");
  fd.write("ok\n");
  fd.close();
  sys.exit(0);

#SO-2019-G:2-N:1126-Cs:13-Pg:1-RC:UL-Title - no header
#SO-2019-G:2-N:0466-Cs:32-Pg:1-RC:UO-Title
if (re.match(r'^SO-2019-G:[0-5]-N:[0-9][0-9][0-9][0-9]-Cs:[0-9][0-9]-Pg:1-RC:[A-Z][A-Z]-Title$', qrcode)):
  print "End for regpage - no header on p1: %s" % (qrcode)
  fd = open("%s.state" % filename, "w");
  fd.write("ok\n");
  fd.close();
  sys.exit(0);

#SO-2019-G:2-N:0466-Cs:32-Pg:17-RC:SI-Dop
if (re.match(r'^SO-2019-G:[0-5]-N:[0-9][0-9][0-9][0-9]-Cs:[0-9][0-9]-Pg:17-RC:[A-Z][A-Z]-Dop$', qrcode)):
  print "End for regpage - no header on p17 (dop): %s" % (qrcode)
  fd = open("%s.state" % filename, "w");
  fd.write("ok\n");
  fd.close();
  sys.exit(0);

#get class
# AO2018-6N1521CS23P9Pr4MP7
# SO-2019-G:2-N:1126-Cs:13-Pg:3-RC:HL-Pr:1-MxPt:5-Problem
res = re.match(r'^SO-2019-G:([0-9])-N:([0-9][0-9][0-9][0-9])-Cs:[0-9][0-9]-Pg:([0-9]+)-RC:[A-Z][A-Z]-Pr:([0-9])-MxPt:([0-9])-Problem$', qrcode)
if (not res):
  print "Whoops no class info in QR: %s" % (qrcode)
  sys.exit(0);

clas = int(res.group(1));
regnumber = int(res.group(2));
page = int(res.group(3));
task = int(res.group(4));
maxpt = int(res.group(5));

print "QR: class: %d, regnumber: %d, page: %d, task: %d, maxPt: %d" % (clas, regnumber, page, task, maxpt);
#if (clas in [3,4]) and (page in [9]):
#  print "End for c:%d p:9 - no header: %s" % (clas,qrcode)
#  fd = open("%s.state" % filename, "w");
#  fd.write("ok\n");
#  fd.close();
#  sys.exit(0);

#if (clas in [5,6,7,8]) and (page in [4,6,8,10,12,14,16,17,18,19,20]):
#  print "End for c:%d p:4/6/8/10etc - no header: %s" % (clas,qrcode)
#  fd = open("%s.state" % filename, "w");
#  fd.write("ok\n");
#  fd.close();
#  sys.exit(0);


grid_items_x = 2 + maxpt;

#if (clas <= 4):
#  grid_items_x = 7;
#elif (clas in [5,6,7,8]):
#  grid_items_x = 9;
#else:
#  print "Whoops: FIXME, need data about maxscore for class: %d" % clas;
#  sys.exit(1)


#SO2018-4N201013P6 - no header
#if (re.match(r'^SO2018[-][3]N[0-9][0-9][0-9][0-9][0-9][0-9]P(6)$', qrcode)):
#  print "End for 3p6 - no header: %s" % (qrcode)
#  fd = open("%s.state" % filename, "w");
#  fd.write("ok\n");
#  fd.close();
#  sys.exit(0);

#if (re.match(r'^SO2018[-][4]N[0-9][0-9][0-9][0-9][0-9][0-9]P(6|8|12)$', qrcode)):
#  print "End for 4p6/6/12 - no header: %s" % (qrcode)
#  fd = open("%s.state" % filename, "w");
#  fd.write("ok\n");
#  fd.close();
#  sys.exit(0);
#
#if (re.match(r'^SO2018[-][5]N[0-9][0-9][0-9][0-9][0-9][0-9]P(6|8|10|12|14)$', qrcode)):
#  print "End for 4p6/6/12 - no header: %s" % (qrcode)
#  fd = open("%s.state" % filename, "w");
#  fd.write("ok\n");
#  fd.close();
#  sys.exit(0);

print "qrdata: %s, rotate: %d" %(qrcode,rotate)

# rotate img + lines
if rotate == 180:
  bw_img = cv2.flip(bw_img, -1); #both axis
  img = cv2.flip(img, -1); #both axis
  h, w, chan = bw_img.shape
  
  rlines = [];
  for (x1,y1,x2,y2) in newlines[0]:
    rlines.append( (w-x2-1, h-y2-1, w-x1-1, h-y1-1) );
    
  newlines[0] = rlines;


#cv2.imshow("hough", cv2.resize(bw_img, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
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
hlines =  sorted(newlines[0], key=lambda x: ( -100*(abs(x[0]-x[2])+abs(x[1]-x[3]))-10*min(x[0],x[2])-min(x[1],x[3]) ) );

linesbylen = {};

prevY = -1
prevLen = -1
prevX = -1
prevY = -1
# group by len
for x1,y1,x2,y2 in hlines:
    phi = math.atan2(x1-x2, y1-y2)*180/np.pi;
#    print phi
    if (phi >160): phi = phi - 180;
    if (phi<-20): phi = phi + 180;
    if (phi<70) or (phi>110): continue;

    cv2.line(img,(x1,y1),(x2,y2),(128,255,128),2)

    curLen = (abs(x1-x2)+abs(y1-y2));
    curX   = min(x1,x2);
    curY   = min(y1,y2);

    if (prevLen == -1):
      prevLen = curLen
    if (prevX == -1):
      prevX = curX
    if (prevY == -1):
      prevY = curY
    
    if (abs(prevLen-curLen)<=55): 
      #samelen
      curLen = prevLen

    if (abs(prevX-curX)<=55):
      #sameX
      curX = prevX

    if (abs(prevY-curY)<=55):
      #sameY
      curY = prevY

    #group by len and X
    key=1000*curLen+10*curX+curY

    if key in linesbylen:
      linesbylen[key].append([x1,y1,x2,y2]);
    else:
      linesbylen[key] = [ [x1,y1,x2,y2] ];
    
    prevLen = curLen
    prevX = curX
    prevY = curY


linesbylen = collections.OrderedDict(sorted(linesbylen.items(), reverse=True));
print linesbylen

#find longest
longest_key = 0;

for key in linesbylen:
  print "key: %s, count: %d" % (key, len(linesbylen[key]))
  if len(linesbylen[key])<1: continue;
  if (int(key) > longest_key): 
    longest_key = int(key);
  
print "longest_lines: key: %d, count: %d" % ( longest_key, len( linesbylen[ longest_key ] ) )
print linesbylen[ longest_key ];

for x1,y1,x2,y2 in linesbylen[ longest_key ]:
  cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)


second_key = 0;
for key in linesbylen:
  if len(linesbylen[key])<3: continue;
  if len(linesbylen[key])>40: continue;
  if (int(key) > second_key and int(key) < int(longest_key*0.85)): 
    second_key = int(key);

if (second_key>0):
  print "second_lines: key: %d, count: %d" % ( second_key, len( linesbylen[ second_key ] ) )
  print "second_lines: ", linesbylen[ second_key ] 
  for x1,y1,x2,y2 in linesbylen[ second_key ]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
else:
  print "second_lines: key: %d, count: %d" % ( second_key, 0 )



third_key = 0;
for key in linesbylen:
  if len(linesbylen[key])<3: continue;
  if (int(key) > third_key and int(key) < second_key): 
    third_key = int(key);

if (third_key>0):
  print "third_lines: key: %d, count: %d" % ( third_key, len( linesbylen[ third_key ] ) )
  print "third_lines: ", linesbylen[ third_key ] 
  for x1,y1,x2,y2 in linesbylen[ third_key ]:
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
else:
  print "third_lines: %d, count: %d" % ( third_key, 0 )



#estimate corners
#left:
_leftX = 0
_rightX = 0;
botB = 0
botT = im_height 
Xc = 0;
# find topmost very-long (>= 95% longest) hline
for key in linesbylen:
  if key < longest_key*0.95: continue;
  for x1,y1,x2,y2 in linesbylen[key]:
    _leftX += min(x1,x2)
    _rightX += max(x1,x2)
    _topY = min(y1,y2)
    _botY = max(y1,y2)
    Xc+=1;
    if (botB < _botY): botB = _botY
    if (botT > _topY): botT = _topY

leftX  = int(round(_leftX/Xc))
rightX = int(round(_rightX/Xc))

longest_len = abs(rightX-leftX);

if longest_len < 0.93*im_width:
  print "too short longest_len: %d of %d Xc:%d" % (longest_len, im_width, Xc);
  longest_len = 0.95*im_width;
  leftX = 0;

print "left: %d, right: %d, bottomTop: %d, bottomBottom: %d, Xc: %d, longest_len: %d" % (leftX, rightX, botT, botB, Xc, longest_len)


#cv2.imshow("hough", img[0:botT, :]);
#cv2.waitKey(0);

topT = 0;
topB = 0;
topL = im_width;
topR = im_width;

Xc = 0
Xkey = 0;

maxClen = -1;
for key in linesbylen:
  if (key > 0.69*longest_len): continue;
  if (key < 0.61*longest_len): continue;
  if maxClen==-1:
    maxClen = len(linesbylen[key]);
    continue;
  if maxClen < len(linesbylen[key]):
    maxClen = len(linesbylen[key]);

#for key in linesbylen:
#  if (key > 0.69*longest_len): continue;
#  if (len(linesbylen[key])<2): continue;
#  print(key, linesbylen[key])
for key in linesbylen:
  if (key > 0.69*longest_len): continue;
  if (key < 0.61*longest_len): continue;
#  if (len(linesbylen[key]) < 0.60*maxClen): continue;
  for line in linesbylen[key]:
    print 'line: ', line, key, maxClen
    if (line[1] < botT*0.85) and (line[1] > botT * 0.4):
      print ' match'
      if (Xkey == 0): Xkey = key;
      if (key != Xkey): continue;
      
      topB += line[1] + line[3];

      if line[0] < line[2]:
        topL = min(topL, line[0]);
        topR = min(topR, line[2]);
      else:
        topL = min(topL, line[2]);
        topR = min(topR, line[0]);
      Xc+=1;
#      print key,line

#fix for erased 2nd line:
if Xc == 0:
  print "bad top block detection: Xc=0"

  Xc = 1;
  ww = longest_len #1459

  topL = leftX
  topR = leftX + int(ww * 0.7)
  
  topB = ww*0.6

#  topB = (botT - 0.145*ww)*2.0;

topL = max( topL-2, 0);
topR = topR+2;

ww = (topR-topL)

topB = int(round( topB / 2.0 / Xc * 1.07));
topT = max( 0, int(topB - ww*0.35)) #0.38


topL = topL + int(0.08*ww) #0.20, 0.13
topR = topR - int(0.12*ww);

#cv2.imshow("hough", img[topT:topB, topL:topR]);
#cv2.waitKey(0);

print "top: (%d,%d) (%d,%d)" % (topL, topT, topR, topB)

res = process_score(bw_img[topT:topB, topL:topR], ww, grid_items_x);

if len(res)==4:
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
      cv2.imwrite("%s_%d_%d.png" % (tmpbase, r,n), imOut);
  
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





