#3
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required(VERSION 2.8.4)

PROJECT(infant_face_detection)

include(../dlib/cmake)

# Tell CMake to compile a program.  We do this with the ADD_EXECUTABLE()
# statement which takes the name of the output executable and then a list of
# .cpp files to compile.  Here each example consists of only one .cpp file but
# in general you will make programs that const of many .cpp files.
ADD_EXECUTABLE(infant_face_detection infant_face_detection.cpp)
# Then we tell it to link with dlib.
TARGET_LINK_LIBRARIES(infant_face_detection dlib)

if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support!")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      #include_directories(${OpenCV_INCLUDE_DIRS})

      #ADD_EXECUTABLE(webcam_face_pose_ex webcam_face_pose_ex.cpp)
      #TARGET_LINK_LIBRARIES(webcam_face_pose_ex dlib ${OpenCV_LIBS} )
      message("OpenCV found!")
   endif()
endif()



