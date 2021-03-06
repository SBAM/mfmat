#!/bin/bash

TPUTRESET=$(tput sgr0 2> /dev/null)
GREEN=$(tput setaf 2 2> /dev/null)
GREY=$(tput setaf 7 2> /dev/null)
RED=$(tput setaf 9 2> /dev/null)
YELLOW=$(tput setaf 11 2> /dev/null)

function green_echo
{
  echo -e $GREEN$@$TPUTRESET
}

function red_echo
{
  echo -e $RED$@$TPUTRESET
}

function grey_echo
{
  echo -e $GREY$@$TPUTRESET
}

function yellow_echo
{
  echo -e $YELLOW$@$TPUTRESET
}

CONFIGURE_LOCATION=$(readlink -f $0)
BASEDIR=$(dirname $CONFIGURE_LOCATION)

export GIT_TAG=${GIT_TAG:-$(git -C $BASEDIR describe --tags --always)}
export GIT_AUTHOR=$(git -C $BASEDIR log -1 | sed -r -n 's/^Author:\s+(.*)/\1/p')
export GIT_COMMIT_DATE=$(git -C $BASEDIR log -1 | sed -r -n 's/^Date:\s+(.*)/\1/p')
export GIT_COMMIT_HASH=$(git -C $BASEDIR log -1 | sed -r -n 's/^commit\s+(.*)/\1/p')

# attempts to determine current OS
function guess_os
{
  SEDX='s/^([[:alpha:]]+)\s+.*release\s+([[:digit:]|[:punct:]]+).*$/\L\1 \2/p'
  if [ -f /etc/redhat-release ]; then
    echo $(sed -r -n "$SEDX" /etc/redhat-release)
  else
    if [ -f /etc/debian_version ]; then
      echo "debian $(cat /etc/debian_version)"
    else
      echo "unknown"
    fi
  fi
}

PLATFORM=$(guess_os)
COMPILER=gcc
LTO=ON

# $1 path
# $N additional flags
function cmake_config
{
  OUTPUT_PATH=$1
  shift
  mkdir -p $OUTPUT_PATH
  if [ 0 -eq $? ]; then
    green_echo "[Will build artifacts in $OUTPUT_PATH]"
  else
    red_echo "[Could not create $OUTPUT_PATH]"
  fi
  pushd > /dev/null $OUTPUT_PATH 2>&1
  grey_echo "  ==> platform=$YELLOW$PLATFORM"
  grey_echo "  ==> compiler=$YELLOW$COMPILER"
  grey_echo "  ==> GIT_AUTHOR=$YELLOW$GIT_AUTHOR"
  grey_echo "  ==> GIT_TAG=$YELLOW$GIT_TAG"
  grey_echo "  ==> GIT_COMMIT_DATE=$YELLOW$GIT_COMMIT_DATE"
  grey_echo "  ==> GIT_COMMIT_HASH=$YELLOW$GIT_COMMIT_HASH"
  grey_echo "  ==> (from $OUTPUT_PATH)"
  grey_echo "  ==> cmake $@ $BASEDIR"
  cmake $@ $BASEDIR
  #cmake --trace $@ $BASEDIR
  popd > /dev/null 2>&1
}

function invoke_cmake
{
  export AMDAPPSDKROOT=/opt/amdgpu-pro
  case $COMPILER in
    clang)
      export CXX=$(which clang++)
      export TOOLCHAIN_FILE=ClangToolchain.cmake ;;
    gcc)
      export CXX=$(which g++)
      export TOOLCHAIN_FILE=GNUToolchain.cmake ;;
  esac
  DEBUGDIR=$BASEDIR/build/Debug
  RELEASEDIR=$BASEDIR/build/Release
  shift
  cmake_config $DEBUGDIR \
               -DCMAKE_BUILD_TYPE=Debug \
               -DLTO=OFF \
               -DCMAKE_TOOLCHAIN_FILE:string=$TOOLCHAIN_FILE \
               $@
  cmake_config $RELEASEDIR \
               -DCMAKE_BUILD_TYPE=Release \
               -DLTO=$LTO \
               -DCMAKE_TOOLCHAIN_FILE:string=$TOOLCHAIN_FILE \
               $@
}

USAGE_STR=\
"usage: $0 \
[-c|--compiler=<gcc|clang>] \
[-l|--lto=<ON|OFF>] \
[-h|--help]"

GETOPT_CMD=\
$(getopt \
    -o c:l:h \
    -l compiler:,lto:,help \
    -n $0 -- $@)

if [ $? -ne 0 ]; then
  red_echo $USAGE_STR && exit 1
fi

while true; do
  case $1 in
    -c|--compiler)
      case $2 in
        gcc|clang) COMPILER=$2 ;;
        *) red_echo $USAGE_STR && exit 1 ;;
      esac; shift 2 ;;
    -l|--lto)
      case $2 in
        ON|OFF) LTO=$2 ;;
        *) usage_die ;;
      esac; shift 2 ;;
    -h|--help)
        green_echo $USAGE_STR && exit 0 ;;
    *) break ;;
  esac
done

invoke_cmake
