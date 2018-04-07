CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_compile_options(-msse4.2)
add_compile_options(-O3)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

FIND_PACKAGE(PCL 1.7 REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${PCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})
ADD_DEFINITIONS(${PCL_DEFINITIONS})


ADD_EXECUTABLE(test_velodyne
  test_velodyne.cpp
)

TARGET_LINK_LIBRARIES(test_velodyne ${PCL_LIBRARIES} ${Boost_LIBRARIES})
                                
add_dependencies(test_velodyne test_velodyne)
