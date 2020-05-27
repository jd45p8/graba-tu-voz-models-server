FROM ubuntu:latest
RUN sudo apt update && sudo apt upgrade
RUN sudo apt install python3.8 python3-virtualenv python3-pip -y