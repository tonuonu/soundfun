cmake_minimum_required(VERSION 3.4)
PROJECT(soundfun)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "-O0 -Wall")
set(CMAKE_CXX_FLAGS "-O0 -Wall")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
FIND_PACKAGE(OpenCV REQUIRED )
find_package(Qt5 COMPONENTS Widgets REQUIRED)

find_library(FFTW_LIBRARY NAMES fftw3 fftw)

INCLUDE_DIRECTORIES(/usr/local/Cellar/fftw/3.3.9/include/ /usr/local/include/)
LINK_DIRECTORIES(/usr/local/Cellar/libsndfile/1.0.31/lib/ /usr/local/lib/)

ADD_EXECUTABLE(soundfun src/main.cpp)
target_link_libraries(soundfun sndfile ${FFTW_LIBRARY} ${OpenCV_LIBS} Qt5::Widgets )

