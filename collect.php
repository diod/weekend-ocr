<?php

$dir = "img";
$fd = opendir($dir);
if (!$fd) die("Unable to opendir");

$imgs = [];
$cnts = [];
while ($row = readdir($fd)) {
  if ($row[0] == '.') continue; //special files
  
  if (preg_match("/^(img-[0-9]+[-][0-9]+)[.]png[.]?(hlines|state|qrdata|score)?$/i",$row,$regs)) {
    if (!empty($regs[2])) {
      $imgs[$regs[1]][$regs[2]]=1;
      @$cnts[$regs[2]]++;
    } elseif (!isset($imgs[$regs[1]])) {
      $imgs[$regs[1]]=[];
    }
  }
}
closedir($fd);
echo("images: ".count($imgs)."\n");
print_r($cnts);

foreach($imgs as $imgname => $info) {
  $info['taskno'] = '';
  if (preg_match("/(img-[0-9]+)[-]([0-9]+)$/i",$imgname,$regs)) {
    $info['pdfname'] = $regs[1];
    $info['page'] = intval($regs[2])+1;
    $info['imgname'] = $imgname;
  }

  if (isset($info['state'])) {
    $info['state'] = trim(file_get_contents( $dir.'/'.$imgname.'.png.state' ));
  } else {
    $info['state'] = '';
  }
  if (isset($info['qrdata'])) {
    $info['qrdata'] = trim(file_get_contents( $dir.'/'.$imgname.'.png.qrdata' ));
    
    if (preg_match("/^R[0-9][0-9][0-9][0-9][0-9][0-9]$/",$info['qrdata'])) {
      $info['state'] = 'regpage';
    }
    if (preg_match("/^SO2018-([0-9])N([0-9][0-9][0-9][0-9][0-9][0-9])P([0-9]+)$/",$info['qrdata'],$xr)) {
      $lvl = intval($xr[1]);
      $pg  = intval($xr[3]);
      
      echo("lvl: $lvl pg: $pg\n");
      
      if ($pg == 1) {
        $info['state'] = 'firstpage';
      } elseif ( ($lvl == 5) && (in_array($pg, [ 6,8,10,12,14 ]) !== false )) {
        $info['state'] = 'no-score';
      } elseif ( ($lvl == 4) && (in_array($pg, [ 6,8,12 ]) !== false )) {
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
    if (preg_match("/^SO2018-([0-9])N([0-9][0-9][0-9][0-9][0-9][0-9])P1$/",$info['qrdata'])) {
      $info['state'] = 'firstpage';
    }
    
    
  } else {
    $info['qrdata'] = '';
  }
  if (isset($info['score'])) {
    $cnt = trim(file_get_contents( $dir.'/'.$imgname.'.png.score' ));
    echo("score: ".$cnt."\n");
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
  $imgs[$imgname] = $info;
}

ksort($imgs);

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





