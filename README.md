##Driverless Car Software of Bkakid team

## Setup Instructions

### Contents
1. [Install Prerequisites](#1-install-prerequisites)
2. [Clone repository](#2-clone-or-fork-repositories)

### 1. Install Prerequisites
1. __Install [Ubuntu 14.04](http://www.ubuntu.com)__
2. __Install required packages__

##install CMake:
$ sudo apt-get install software-properties-common
$ sudo add-apt-repository ppa:george-edison55/cmake-3.x
$ sudo apt-get update
$ sudo apt-get install cmake

**********************************
##install Astra Camera Driver and OpenNI2:
use OpenNI-Linux-Arm-2.3
run install.sh to generate OpenNIDevEnvironment, which contains OpenNI development environment 

$ sudo chmod a+x install.sh
$ sudo ./install.sh

please replug in the device for usb-register
add environment variables.

$ source OpenNIDevEnvironment


********************************************
##install I2C For JetsonTK1

$ sudo apt-get install -y i2c-tools
$ apt-cache policy i2c-tools

********************************************
##install GIT

$ sudo apt-get install git

********************************************
##install PWM Servo Driver Board 

$ sudo apt-get install libi2c-dev i2c-tools
$ sudo i2cdetect -y -r 1

#JHPWMDriver Install
Set time and date first
  



