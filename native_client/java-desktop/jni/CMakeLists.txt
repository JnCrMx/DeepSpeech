# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
             deepspeech-jni

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             deepspeech_wrap.cpp )

add_library( deepspeech-lib
             SHARED
             IMPORTED )

set_target_properties( deepspeech-lib
             PROPERTIES
             IMPORTED_LOCATION
             ${CMAKE_SOURCE_DIR}/libs/libdeepspeech.so )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       deepspeech-jni

                       deepspeech-lib )

add_custom_command( TARGET deepspeech-jni POST_BUILD
             COMMAND ${CMAKE_COMMAND} -E copy
             ${CMAKE_SOURCE_DIR}/libs/libdeepspeech.so
             ${CMAKE_SOURCE_DIR}/../libdeepspeech/src/main/resources/libdeepspeech.so )
add_custom_command( TARGET deepspeech-jni POST_BUILD
             COMMAND ${CMAKE_COMMAND} -E copy
             $<TARGET_FILE:deepspeech-jni>
             ${CMAKE_SOURCE_DIR}/../libdeepspeech/src/main/resources/libdeepspeech-jni.so )

