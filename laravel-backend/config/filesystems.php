<?php
return ['default'=>env('FILESYSTEM_DISK','local'),'disks'=>[
 'local'=>['driver'=>'local','root'=>storage_path('app/private'),'serve'=>true,'throw'=>false],
 'doctors'=>['driver'=>'local','root'=>storage_path('app/doctors'),'serve'=>false,'throw'=>false],
 ],'links'=>[public_path('storage')=>storage_path('app/public')]];
