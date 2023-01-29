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
SHARED=False
LTO=True

# $1 build type (Debug/Release)
function configure_build
{
  OUTPUT_PATH=$BASEDIR/build/$1
  grey_echo "  ==> platform=$YELLOW$PLATFORM"
  grey_echo "  ==> compiler=$YELLOW$COMPILER"
  grey_echo "  ==> SHARED=$YELLOW$SHARED"
  grey_echo "  ==> LINK_TIME_OPTIMIZATION=$YELLOW$LTO"
  conan install $BASEDIR \
        -b missing \
        -pr $BASEDIR/conan/profiles/$2 \
        -s build_type=$1 \
        -if $OUTPUT_PATH \
        -o *:shared=$SHARED \
        -o *:lto=$LTO
  conan build $BASEDIR -bf $OUTPUT_PATH -c
}

function setup_fedora
{
  case $COMPILER in
    clang)
      configure_build Debug fedora_clang
      configure_build Release fedora_clang ;;
    gcc)
      configure_build Debug fedora_gcc
      configure_build Release fedora_gcc ;;
    *)
      red_echo "Unsupported compiler=$COMPILER on platform=$PLATFORM"
      usage_die ;;
  esac
}

USAGE_STR=\
"usage: $0 \
[-p|--platform <fedora>] \
[-c|--compiler <gcc|clang>] \
[-s|--shared <True|False>] \
[-l|--lto <True|False>]"

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
    -o p:c:s:l:h \
    -l platform:,compiler:,shared:,lto:,help \
    -n $0 -- $@)

if [ $? -ne 0 ]; then
  usage_die
fi

while true; do
  case $1 in
    -p|--platform)
      case $2 in
        fedora*) PLATFORM=$2 ;;
        *) usage_die ;;
      esac; shift 2 ;;
    -c|--compiler)
      case $2 in
        gcc|clang) COMPILER=$2 ;;
        *) usage_die ;;
      esac; shift 2 ;;
    -s|--shared)
      case $2 in
        True|False) SHARED=$2 ;;
        *) usage_die ;;
      esac; shift 2 ;;
    -l|--lto)
      case $2 in
        True|False) LTO=$2 ;;
        *) usage_die ;;
      esac; shift 2 ;;
    -h|--help) usage_help ;;
    *) break ;;
  esac
done

case $PLATFORM in
  fedora*) setup_fedora ;;
  *) red_echo "Unsupported platform=$PLATFORM"; usage_die ;;
esac
