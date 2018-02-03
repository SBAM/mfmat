#!/bin/sh

BUILD_DIR=build

if [ -d $BUILD_DIR ]; then
    cd $BUILD_DIR
    if [ -f Makefile ]; then
        make clean
    fi
else
    mkdir build
    cd build
fi

cmake ..
