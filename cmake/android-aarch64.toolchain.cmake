# gap? wire NDK path in CI — placeholder Android aarch64 toolchain file.
# Usage:
#   cmake -B build-android -DCMAKE_TOOLCHAIN_FILE=cmake/android-aarch64.toolchain.cmake \
#     -DANDROID_NDK=$ANDROID_NDK -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 28)
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
if(DEFINED ENV{ANDROID_NDK})
  set(CMAKE_ANDROID_NDK $ENV{ANDROID_NDK})
endif()
# gap? validate shared-lib + JNI packaging for welvet.cpp on Android
