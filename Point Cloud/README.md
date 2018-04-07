# Point-Cloud

Convert .txt file (Velodyne KITTI Data) to pcd format.

Command:
```
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ ./test_velodyne ../2011_09_26/2011_09_26_drive_0001_extract/velodyne_points/data/0000000000.txt 
```

To view the pcd data, run:
```
$ pcl_viewer --multiview test_pcd1.pcd test_pcd.pcd 
```
