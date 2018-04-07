#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <fstream>
#include <string>
using namespace std;
int main (int argc, char** argv)
{

    pcl::PointCloud<pcl::PointXYZI> cloud;
    std::ifstream input;
  //  input.open("2011_09_26/2011_09_26_drive_0001_extract/velodyne_points/data/000000000" + to_string(0) + ".txt");
    if (argc >= 2)
    {
      input.open(argv[1]);
    }
    if(!input.good())
    {
        std::cerr<<"Cannot open file : "<<std::endl;
        return 0;
    }

    cloud.clear();
    cloud.height = 1;
    std::string s1;


    while(!std::getline(input, s1, ' ').eof())
    {
      std::string s2,s3,s4;
      pcl::PointXYZI point;

       float f = std::stof(s1);
    //   std::cout << "f" << f << '\n';
       point.x = f;
      std::getline(input, s2, ' ');
      f = std::stof(s2);
    //  std::cout << "f" << f << '\n';
      point.y = f;
      std::getline(input, s3, ' ');
      f = std::stof(s3);
    //  std::cout << "f" << f << '\n';
      point.z = f;
      std::getline(input, s4);
      f = std::stof(s4);
    //  std::cout << "f" << f << '\n';
      point.intensity = f;
      // std::cout << "  pointtttttttttttttttttttttttttttt  " << point.x  <<" " <<point.y<< std::endl;

      cloud.push_back(point);
    }

  pcl::io::savePCDFileASCII ("test_pcd1.pcd", cloud);
  std::cerr << "Saved " << cloud.points.size () << " data points to test_pcd.pcd." << std::endl;

  for (size_t i = 0; i < cloud.points.size (); ++i)
    std::cerr << "    " << cloud.points[i].x << " " << cloud.points[i].y << " " << cloud.points[i].z << std::endl;

  return (0);
}
