# Deep-Learning-for-Point-Cloud
# NARF 
Extracts NARF Features from Point cloud without range data.

Command:
```
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ ./narf_feature_extraction test.pcd
```
While running the executable, test.pcd can be any .pcd file on which you want to extract features.

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
