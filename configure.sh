#!/bin/sh

TPUTRESET=`tput sgr0`
GREEN=`tput setaf 2`
RED=`tput setaf 1`

function green_echo
{
    echo -e "$GREEN$@$TPUTRESET"
}

function red_echo
{
    echo "$RED$@$TPUTRESET"
}

ORIGDIR=`pwd`
BASEDIR=$ORIGDIR/$(dirname $0)

# $1 path
# $2 Debug|Release
function cmake_config
{
    mkdir -p $1
    if [ 0 -eq $? ]; then
        green_echo "[Will build $2 artifacts in $1]"
    else
        red_echo "[Could not create $1]"
        exit 1
    fi
    pushd > /dev/null $1 2>&1
    green_echo "  ==> (from `pwd`) cmake -DCMAKE_BUILD_TYPE=$2 $BASEDIR"
    cmake -DCMAKE_BUILD_TYPE=$2 $BASEDIR
    popd > /dev/null 2>&1
}

DEBUGDIR=$BASEDIR/build/Debug
RELEASEDIR=$BASEDIR/build/Release
cmake_config $DEBUGDIR "Debug"
cmake_config $RELEASEDIR "Release"
