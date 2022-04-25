cmake_minimum_required(VERSION 3.10)
project(KalmanFilter)
find_package(Eigen3 QUIET)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
  # Possibly map additional variables to the EIGEN3_ prefix.
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(KalmanFilter KalmanFilter.cpp)
