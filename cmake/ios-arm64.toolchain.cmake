# gap? wire Xcode / ios.toolchain.cmake — placeholder iOS arm64 file.
# Usage (on macOS with Xcode):
#   cmake -B build-ios -GXcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64
set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")
# gap? produce .framework / static lib for iOS + validate C ABI
