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
# $2 Debug/Release
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

function launch
{
  DEBUGDIR=$BASEDIR/build/Debug
  RELEASEDIR=$BASEDIR/build/Release
  shift
  cmake_config $DEBUGDIR \
               -DCMAKE_BUILD_TYPE=Debug \
               -DLTO=$LTO \
               $@
  cmake_config $RELEASEDIR \
               -DCMAKE_BUILD_TYPE=Release \
               -DLTO=$LTO \
               $@
}

function setup_fedora
{
  export THIRD_PARTIES_SYS=/usr
  case $COMPILER in
    clang)
      export CC=$THIRD_PARTIES_SYS/bin/clang
      export CXX=$THIRD_PARTIES_SYS/bin/clang++
      export CMAKE_AR=$THIRD_PARTIES_SYS/bin/llvm-ar
      export CMAKE_NM=$THIRD_PARTIES_SYS/bin/llvm-nm
      export CMAKE_RANLIB=$THIRD_PARTIES_SYS/bin/llvm-ranlib ;;
    gcc)
      export CC=$THIRD_PARTIES_SYS/bin/gcc
      export CXX=$THIRD_PARTIES_SYS/bin/g++
      export CMAKE_AR=$THIRD_PARTIES_SYS/bin/gcc-ar
      export CMAKE_NM=$THIRD_PARTIES_SYS/bin/gcc-nm
      export CMAKE_RANLIB=$THIRD_PARTIES_SYS/bin/gcc-ranlib ;;
  esac
  export BOOST_ROOT=$THIRD_PARTIES_SYS
  launch $@
}

function setup_centos
{
  export THIRD_PARTIES_SYS=/usr/local
  export AMDAPPSDKROOT=/opt/amdgpu-pro
  case $COMPILER in
    gcc)
      export CC=$THIRD_PARTIES_SYS/bin/gcc
      export CXX=$THIRD_PARTIES_SYS/bin/g++
      export CMAKE_AR=$THIRD_PARTIES_SYS/bin/gcc-ar
      export CMAKE_NM=$THIRD_PARTIES_SYS/bin/gcc-nm
      export CMAKE_RANLIB=$THIRD_PARTIES_SYS/bin/gcc-ranlib ;;
    *)
      red_echo "Unsupported compiler=$COMPILER on platform=$PLATFORM"
      usage_die ;;
  esac
  export BOOST_ROOT=$THIRD_PARTIES_SYS
  launch $@
}

USAGE_STR=\
"usage: $0 \
[-p|--platform=<fedora|centos>] \
[-c|--compiler=<gcc|clang>] \
[-l|--lto=<ON|OFF>] \
[-h|--help]"

function usage_die()
{
  red_echo $USAGE_STR
  exit 1
}

function usage_help()
{
  green_echo $USAGE_STR
  exit 0
}

GETOPT_CMD=\
$(getopt \
    -o p:c:l:h \
    -l platform:,compiler:,lto:,help \
    -n $0 -- $@)

if [ $? -ne 0 ]; then
  usage_die
fi

while true; do
  case $1 in
    -p|--platform)
      case $2 in
        centos*|fedora*) PLATFORM=$2 ;;
        *) usage_die ;;
      esac; shift 2 ;;
    -c|--compiler)
      case $2 in
        gcc|clang|analyzer) COMPILER=$2 ;;
        *) usage_die ;;
      esac; shift 2 ;;
    -l|--lto)
      case $2 in
        ON|OFF) LTO=$2 ;;
        *) usage_die ;;
      esac; shift 2 ;;
    -h|--help) usage_help ;;
    *) break ;;
  esac
done

case $PLATFORM in
  centos*) setup_centos ;;
  fedora*) setup_fedora ;;
  *) red_echo "Unsupported platform=$PLATFORM"; usage_die ;;
esac
