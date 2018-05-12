<?php

#24apr 424140434
#07may 507154350

$dir = "img";
$fd = opendir($dir);
if (!$fd) die("Unable to opendir");

$imgs = [];
$cnts = [];
while ($row = readdir($fd)) {
  if ($row[0] == '.') continue; //special files
  
  if (preg_match("/^(img-[0-9]+|Scan.*)[-]([0-9]+)[.]png[.]?(hlines|state|qrdata|score)?$/i",$row,$regs)) {
    $pdfname = $regs[1].'-'.$regs[2];
    $suffix = @$regs[3];
    if (!empty($suffix)) {
      $imgs[$pdfname][$suffix]=1;
      @$cnts[$suffix]++;
    } elseif (!isset($imgs[$pdfname])) {
      $imgs[$pdfname]=[];
    }
  } else {
    echo("non-matching filename: ".$row."\n");
  }
}
closedir($fd);
echo("images: ".count($imgs)." total (png)\n");
print_r($cnts);

foreach($imgs as $imgname => $info) {
  $info['taskno'] = '';
  if (preg_match("/(img-[0-9]+|Scan.*)[-]([0-9]+)$/i",$imgname,$regs)) {
    $info['pdfname'] = $regs[1];
    $info['page'] = intval($regs[2])+1;
    $info['imgname'] = $imgname;
  }

  $info['pdfts'] = get_pdfts($info['pdfname']);

  if (isset($info['state'])) {
    $info['state'] = trim(file_get_contents( $dir.'/'.$imgname.'.png.state' ));
  } else {
    $info['state'] = '';
  }

  if (isset($info['qrdata'])) {
    $qrdata = explode(' ', file_get_contents( $dir.'/'.$imgname.'.png.qrdata'));
    $info['qrdata'] = trim($qrdata[0]);
    
    if (isset($qrdata[1])) $info['rotate'] = intval($qrdata[1]);
    else $info['rotate']=0;
    
    if (preg_match("/^R[0-9][0-9][0-9][0-9][0-9][0-9]$/",$info['qrdata'])) {
      $info['state'] = 'regpage';
    }
    if (preg_match("/^SO2018-([0-9])N([0-9][0-9][0-9][0-9][0-9][0-9])P([0-9]+)$/",$info['qrdata'],$xr)) {
      $lvl = intval($xr[1]);
      $pg  = intval($xr[3]);
      
//      echo("lvl: $lvl pg: $pg\n");
      
      if ($pg == 1) {
        $info['state'] = 'firstpage';
      } elseif ( ($lvl == 5) && (in_array($pg, [ 6,8,10,12,14 ]) !== false )) {
        $info['state'] = 'no-score';
      } elseif ( ($lvl == 4) && (in_array($pg, [ 6,8,12 ]) !== false )) {
        $info['state'] = 'no-score';
      } elseif ( ($lvl == 3) && (in_array($pg, [ 6 ]) !== false )) {
        $info['state'] = 'no-score';
      }
      // lvl=3 we have no data for now
      // lvl=2 have no no-score pages
      // lvl=1 have no no-score pages

      $taskno = '';
      
      if ($lvl == 1) {
        switch ($pg) {
          case 3: $taskno = 1; break;
          case 4: $taskno = 2; break;
          case 5: $taskno = 3; break;
          case 6: $taskno = 4; break;
          case 7: $taskno = 5; break;
          case 8: $taskno = 6; break;
        }
      }
      if ($lvl == 2) {
        switch ($pg) {
          case 3: $taskno = 1; break;
          case 4: $taskno = 2; break;
          case 5: $taskno = 3; break;
          case 6: $taskno = 4; break;
          case 7: $taskno = 5; break;
          case 8: $taskno = 6; break;
          case 9: $taskno = 7; break;
        }
      }
      if ($lvl == 3) {
        switch ($pg) {
          case 3: $taskno = 1; break;
          case 4: $taskno = 2; break;
          case 5: $taskno = 3; break;
          case 7: $taskno = 4; break;
          case 8: $taskno = 5; break;
          case 9: $taskno = 6; break;
          case 10: $taskno = 7; break;
        }
      }

      if ($lvl == 4) {
        switch ($pg) {
          case 3: $taskno = 1; break;
          case 4: $taskno = 2; break;
          case 5: $taskno = 3; break;
          case 7: $taskno = 4; break;
          case 9: $taskno = 5; break;
          case 10: $taskno = 6; break;
          case 11: $taskno = 7; break;
        }
      }

      if ($lvl == 5) {
        switch ($pg) {
          case 3: $taskno = 1; break;
          case 4: $taskno = 2; break;
          case 5: $taskno = 3; break;
          case 7: $taskno = 4; break;
          case 9: $taskno = 5; break;
          case 11: $taskno = 6; break;
          case 13: $taskno = 7; break;
        }
      }

      $info['taskno'] = $taskno;      
    }

    //fix
    if ($info['state'] == 'no-qr-found' || $info['state'] == 'empty nohlines') {
      $info['state'] = '';
    }    
    
  } else {
    $info['qrdata'] = '';
  }
  
  if (isset($info['score'])) {
    $cnt = trim(file_get_contents( $dir.'/'.$imgname.'.png.score' ));
//    echo("score: ".$cnt."\n");
    $sc = explode(' ', $cnt);
    
    if (count($sc) == 4) {
      $info['score_1'] = $sc[0];
      $info['score_2'] = $sc[1];
      $info['score_3'] = $sc[2];
      $info['score_4'] = $sc[3];

      if (($sc[0] > -1) && ($sc[1] > -1)) {
        $info['state'] = 'ok';
      }
    } else {
      $info['score_1'] = '?';
      $info['score_2'] = '?';
      $info['score_3'] = '?';
      $info['score_4'] = '?';
    }
  } elseif ($info['state'] == 'regpage' || $info['state'] == 'firstpage' || $info['state'] == 'no-score' || $info['state'] == 'empty nohlines') {
    $info['score_1'] = '';
    $info['score_2'] = '';
    $info['score_3'] = '';
    $info['score_4'] = '';
  } else {
    $info['score_1'] = '?';
    $info['score_2'] = '?';
    $info['score_3'] = '?';
    $info['score_4'] = '?';
  }

  @$cnts[$info['state']]++;
  $imgs[$imgname] = $info;
}

print_r($cnts);

uasort($imgs,function ($a,$b) {
  if ($a['pdfts'] > $b['pdfts']) return 1;
  if ($a['pdfts'] < $b['pdfts']) return -1;
  return strcmp($a['imgname'],$b['imgname']);
});

#print_r($imgs);
#ksort($imgs);

$fd = fopen("list.csv","w");
$hdr = [ 'pdfname', 'page', 'imgname', 'state', 'qrdata', 'taskno', 'score_1', 'score_2','score_3', 'score_4' ];
fwrite($fd, implode(';', $hdr)."\n");

foreach($imgs as $imgname => $info) {
  $row = [];
  foreach($hdr as $k) {
    $row[] = $info[$k];
  }
  fwrite($fd, implode(';', $row)."\n");
}
fclose ($fd);


$fd = fopen("vo2018-kirill.csv","r");
fgets($fd);

$students = [];
$bcode2rc = [];
$rc2bcode = [];
while ($row = fgetcsv($fd, 4096)) {
  $student = [
    'fio' => $row[0],
    'regid' => intval($row[1]),
    'level' => intval($row[2]),
    'room'  => $row[3],
    'protocol' => intval($row[4]),
    'ts'    => $row[5],
    'bcode' => intval($row[6]),
  ];
  $students[$student['regid']] = $student;
  $bcode2rc[$student['bcode']] = $student['regid'];
  $rc2bcode[$student['regid']][$student['bcode']] = 1;
}
fclose($fd);

$students[652] = [
  'fio' => 'Мовсисян Никита',
  'regid' => 652,
  'level' => 1,
  'bcode' => 1043205,
  'ts' => ''
];
$bcode2rc[1042305] = 652;

echo("kdata: ".count($students).", b2r: ".count($bcode2rc)." rc2bcode: ".count($rc2bcode)."\n");

$works = [];
foreach($imgs as $info) {
  if (!empty($info['qrdata'])) {
    if (preg_match("/^R([0-9])([0-9][0-9][0-9][0-9])([0-9])$/",$info['qrdata'],$regs)) {
      //regpage
      $rc = intval($regs[2]);

      if (!isset($students[$rc])) {
        echo("Unknown students reglist: ".$rc."\n");
      }      
      $students[$rc]['reglist'][] = $info['pdfname'].'_p'.$info['page'];
      continue;
    }
    if (preg_match("/^SO2018-([0-9])N([0-9][0-9][0-9][0-9][0-9][0-9])P([0-9]+)$/",$info['qrdata'],$xr)) {
      $bcode = intval($xr[1])*1000000 + intval($xr[2]);
      $page = intval($xr[3]);
      
      if (!isset($bcode2rc[$bcode])) {
        echo("No rc for bcode: ".$bcode."\n");
        print_r($info);
        continue;
      }
      
      $rc = $bcode2rc[$bcode];
      $rc2bcode[$rc][$bcode] = 1;
      
      $works[$bcode]['rc'] = $rc;
      $works[$bcode]['lvl'] = intval($xr[1]);
      if (isset($works[$bcode]['pages'][$page])) {
        echo("Overwrite $bcode/$page (".$works[$bcode]['pages'][$page].") with ".$info['imgname']."\n");
      }
      @$works[$bcode]['pages'][$page] = $info['imgname'];
      if ($info['taskno']) {
        $works[$bcode]['tasks'][$info['taskno']] = [ $info['score_1'], $info['score_2'], $info['score_3'] ];
        $works[$bcode]['taskpage'][$info['taskno']] = $info['pdfname'].'_p'.$info['page'];
      }
      $works[$bcode]['page'][$page] = $info['page'];
    }
  }
}

echo("works: ".count($works)."\n");


$cnt = [];
foreach($works as $bcode => $work) {
  $have_2checks = 0;
  $have_missing = 0;
  for($i=1;$i<8;$i++) {
    $checks = 0;
    $missing = 0;
    for($j=1;$j<4;$j++) {
      $v = @$work['tasks'][$i][ $j-1 ];
      if (($v === null) || ($v === '?')) {
        $missing++;
      } elseif ($v>=0) {
        $checks++;
      }
      $row[] = $v;
    }
    if ($checks >= 2) $have_2checks++;
    if ($missing > 0) $have_missing++;
  }

  if ($work['lvl'] == 1) {
    $have_missing--;
    $have_2checks++;
  }  

  if ($have_2checks == 7) {
    $re = 'complete';
  } elseif (($have_missing == 7) || ( ($work['lvl'] == 1)&&($have_missing == 6) ) ){
    $re = 'empty';
    print_r($work);
  } else if ($have_missing) {
    $re = 'missing tasks';
  } else {
    $re = 'no 2 checks';
  }
  
  $works[$bcode]['status'] = $re;
  @$cnt[$re]++;  
}

print_r($cnt);



$fd = fopen("result.csv", "w");
$cols = [ 'штамп времени', 'код условия', 'регкод'];
for($i=1;$i<8;$i++) {
  for($j=1;$j<4;$j++) $cols[] = $i.','.$j;
}
$cols = array_merge( $cols, [ 'статус', 'фио', 'регбланк1', 'регбланк2' ]);
for($i=1;$i<8;$i++) $cols[] = 'задача '.$i;
$cols[] = 'комментарий';
for($i=1;$i<15;$i++) $cols[] = 'страница '.$i;

fwrite($fd, implode(';', $cols)."\n");


foreach($rc2bcode as $rc => $bb) {
  $st = $students[$rc];
  if (!is_array($bb)) {
    $bb = [ 0 => 0 ]; //dummy empty work
  }
  foreach($bb as $bcode => $dummy) {
    if (isset($works[$bcode])) {
      $work = $works[$bcode];
    } else {
      $work = [ 'status' => 'no-work' ];
    }

    $row = [ $st['ts'], $st['bcode'], $st['regid'] ];

    $level = $st['level'];

    for($i=1;$i<8;$i++) {
      for($j=1;$j<4;$j++) {
        $v = @$work['tasks'][$i][ $j-1 ];
        $row[] = $v;
      }
    }

    $row[] = $work['status'];

    $row[] = $st['fio'];
    $row[] = @$st['reglist'][0]; //img0
    if (isset($st['reglist'][1])) {
      $row[] = @$st['reglist'][1]; //regblank
    } else {
      $row[] = '';
    }
  

    $missing_taskpage=[];  
    for($i=1;$i<8;$i++) {
      if (isset($work['taskpage'][$i])) {
        $row[] = $work['taskpage'][$i];
      } else {
        $row[] = '';
      }
    }
  
  
  
    $row[] = '';
    for($i=1;$i<14;$i++) {
      $row[] = @$work['page'][$i];
    }

    fwrite($fd, implode(';', $row)."\n");
  }
}
fclose ($fd);

function get_pdfts($pdfname) {
  if (preg_match("/^img-([0-9]?[0-9])([0-3][0-9])([0-9][0-9])([0-9][0-9])([0-9][0-9])$/",$pdfname, $regs)) {
    $tm = mktime( intval($regs[3]), intval($regs[4]), intval($regs[5]), intval($regs[1]), intval($regs[2]) );
    return $tm;
  }
  if (preg_match("/^Scanned[-]image_([0-9][0-9])[-]([0-9][0-9])[-](20[0-9][0-9])[-]([0-9][0-9])([0-9][0-9])([0-9][0-9])/",$pdfname, $regs)) {
    $tm = mktime( intval($regs[4]), intval($regs[5]), intval($regs[6]), intval($regs[2]), intval($regs[1]), intval($regs[3]) );
//    echo($pdfname.' '.date('Y-m-d H:i:s',$tm)."\n");
    return $tm;
  }
  print_r($pdfname."\n");
  return 0;
}

