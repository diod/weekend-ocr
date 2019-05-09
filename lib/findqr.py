#!/usr/bin/python

import cv2
import numpy as np

#
# estimate image orientation by QR
#

def estimate_img_rot(orig_img):
  # orig img
  im_height, im_width, im_channels = orig_img.shape
  
  if (im_width > im_height):
    scale = float(im_width)/320;
  else:
    scale = float(im_height)/320;

  small_img = cv2.resize(orig_img, (int(im_width/scale), int(im_height/scale)))
  sm_height, sm_width, sm_channels = small_img.shape

  rot,qrpos = _estimate_small_img_rot(cv2.cvtColor(small_img,cv2.COLOR_BGR2GRAY));
  if rot == 90:
    orig_img = np.transpose(orig_img, (1,0,2));
    orig_img = cv2.flip(orig_img, 1);
  if rot == 180:
    orig_img = cv2.flip(orig_img, 0);
    orig_img = cv2.flip(orig_img, 1);
  if rot == -90:
    orig_img = np.transpose(orig_img, (1,0,2));
    orig_img = cv2.flip(orig_img, 0);

  wh = max(qrpos[2]-qrpos[0], qrpos[3]-qrpos[1])*scale*0.07;

  # x1 y1 x2 y2
  qrpos = (
    int(qrpos[0]*scale - wh),
    int(qrpos[1]*scale - wh),
    int(qrpos[2]*scale + wh),
    int(qrpos[3]*scale + wh))

  print "estimate_orientation_small: %dx%d, rot: %d, qrpos: (%d %d) (%d %d)" % ( sm_width, sm_height, rot, qrpos[0], qrpos[1], qrpos[2], qrpos[3] );

  return [orig_img, qrpos];


def _estimate_small_img_rot(bwimg):
  height, width = bwimg.shape

  (_, bw) = cv2.threshold(bwimg,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);
  bw = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6)), 1);
  bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1);

  cv2.imshow("hough", cv2.resize(bw, None, fx=3,fy=3, interpolation=cv2.INTER_NEAREST));
  cv2.waitKey(0)

  proj_x = np.sum(bw, axis=0, dtype=np.uint32);
  proj_y = np.sum(bw, axis=1, dtype=np.uint32);
  
  proj_sx = np.sort(proj_x);
  proj_sy = np.sort(proj_y);
  
  med_x = proj_sx[int(0.8*width)];
  med_y = proj_sy[int(0.8*height)];
  
  #print "med_x: ", med_x, "med_y: ", med_y;
  
  non_empty_columns = np.where( proj_x > med_x )[0]
  non_empty_rows    = np.where( proj_y > med_y )[0]

  print non_empty_columns;
  print non_empty_rows;

  if (len(non_empty_rows) <= 20) or (len(non_empty_columns) <= 20):
    print "%s: no qr found?" % sys.argv[1]
    sys.exit(0);

  x0 = non_empty_columns[0];
  px = x0;
  stepx=0;
  for x in non_empty_columns:
#    print "Lx: %d x0: %d, x: %d, px: %d, stepx: %d" % (Lx, x0, x, px, stepx)
    if (x-px>12): # step
      if stepx>20:
        break;
      x0 = x;
      stepx = 0;
    else:
      stepx += 1;
    px = x;  

  Lx = x0;

  x0 = max(non_empty_columns);
  px = x0;
  stepx=0;
  for x in reversed(non_empty_columns):
    if (px-x>12): # step
      if (stepx>20):
        break;
      x0 = x;
      stepx=0;
    else:
      stepx+=1;
    px = x;  
  
  Rx = x0;

  
  y0 = non_empty_rows[0];
  py = y0;
  stepy=0;
  for y in non_empty_rows:
    if (y-py>12): # step
      if stepy>20:
        break;
      y0 = y;
      stepy=0;
    else:
      stepy+=1;
    py = y;  

  Ty = y0;

  
  y0 = max(non_empty_rows);
  py = y0;
  stepy = 0;
  for y in reversed(non_empty_rows):
    if (py-y>12): # step
      if stepy>20:
        break;
      y0 = y;
      stepy=0;
    else: 
      stepy+=1;
    py = y;  
  
  By = y0;


  rot = 0;
  #quadrants
  if (Lx*2 > width):
    #right
    if (Ty*2 > height):
      #bottom;
      rot = -90;
      qrpos = ( Ty, width-Rx , By, width-Lx );
    else:
      #top
      rot = 0;
      qrpos = ( Lx, Ty, Rx, By );
  else:
    #left
    if (Ty*2 > height):
      #bottom;
      rot = 180;
      qrpos = ( width-Rx, height-By, width-Lx, height-Ty );
    else:
      #top
      rot = 90;
      qrpos = ( Ty, width-Rx , By, width-Lx );
    
  print "lX: ", Lx,", rX: ",Rx, ", tY: ",Ty, ", bY: ", By, " -> rot: ", rot
  return [rot,qrpos];

#
# get image cropbox
#

def get_image_cropBox(bwimg, padding=20):
  rows, cols = bwimg.shape

  non_empty_columns = np.where( np.sum(bwimg, axis=0, dtype=np.uint32) > 5000 )[0]
  non_empty_rows    = np.where( np.sum(bwimg, axis=1, dtype=np.uint32) > 5000 )[0]

  if (len(non_empty_rows) <= 100) or (len(non_empty_columns) <= 100):
    print "%s: empty nohlines" % sys.argv[1]
    fd = open("%s.state" % sys.argv[1], "w");
    fd.write("empty nohlines\n");
    fd.close();
    sys.exit(0);

  leftX = non_empty_columns[0];
  l20 = non_empty_columns[20];
  
  topY  = non_empty_rows[0];
  t20 = non_empty_rows[20];

  if (l20 - leftX > 50): #remove a gapped noise
    for n in range(1,20):
      if (non_empty_columns[n] - leftX > 30):
        leftX = non_empty_columns[n];
        break;

  if (t20 - topY > 50): #remove a gapped noise
    for n in range(1,20):
      if (non_empty_rows[n] - topY > 30):
        topY = non_empty_rows[n];
        break;

  #x1:x2,y1:y2
  
  cropBox = ( max(0, topY - padding),
              min(max(non_empty_rows) + padding, rows),
              max(0, leftX - padding),
              min(max(non_empty_columns) + padding, cols))

  print "Crop box: (%d,%d) - (%d,%d)" %(cropBox[0], cropBox[2], cropBox[1], cropBox[3]);
  return cropBox


#
# hlines/get page rot angle (fine)
#

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
  lines= cv2.HoughLinesP(image=dilate,rho=1,theta=np.pi/1440,threshold=600,minLineLength=min_line,maxLineGap=7);
  if lines is None:
    print "get_angle: Looks like empty/invalid page -- no long lines here"
    return [ 0.0, [], cropBox ];

  print "get_angle: Hough lines: %d" % (len(lines[0]))

  avgangle = 0.0
  avgcnt = 0

  angleG = {}
  for n in range(0,len(lines)):
    for x1,y1,x2,y2 in lines[n]:
      cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

      phi = math.atan2(x1-x2, y1-y2)*180/np.pi;
      if (phi<0):
        phi+=180.0

      #horizontal lines    
#      if (phi>70 and phi<110):
      qphi = int( (phi+2.5) / 5.0) * 5;
      if (str(qphi) in angleG):
        angleG[str(qphi)].append(phi);
      else:
        angleG[str(qphi)] = [];
        angleG[str(qphi)].append(phi);
    
      print "phi: %.1f (%d %d),(%d,%d)" % (phi, x1,y1, x2,y2)

#  print angleG
  maxlenQphi = -1
  for qphi,lst in angleG.iteritems():
    if (maxlenQphi == -1):
      maxlenQphi = qphi
      continue;
    if (len(angleG[maxlenQphi]) < len(lst)):
      maxlenQphi = qphi

#  print "maxlenQphi: ", maxlenQphi
  avgcnt = len(angleG[maxlenQphi])
  avgangle = sum(angleG[maxlenQphi])

  if (avgcnt == 0):
    print "get_angle: Looks like empty/invalid page -- no long lines here"
    return [ 0.0, [], cropBox ];

  angle = -(avgangle / avgcnt) + 90.0;

  if (angle>45): angle=angle - 90.0;

  print "get_angle: avg: cnt=%d anglediff = %0.4f" % (avgcnt, angle)

#  cv2.imshow("dilate", cv2.resize(dilate, None, fx=0.5,fy=0.5, interpolation=cv2.INTER_NEAREST));
#  cv2.waitKey(0);

  return [angle,lines,cropBox]

