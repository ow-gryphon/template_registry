# Docker Directory
This directory contains Dockerfile and docker-compose files for different set
ups, these can be differentiated by appending an appropriate extension such as
Dockerfile.mgti for the MGTI platform.

### Getting Started
1. Ensure you have the docker environment set up. For Windows users, OW recommends using our [Vagrant-backed  solution](https://bitbucket.org/oliverwymantechssg/labs-vagrant-docker-host/overview)
2. [Optional] From the project root, run `docker-compose -f docker/docker-compose.yml build` to build the image right now.
3. From the project root, run `docker-compose -f docker/docker-compose.yml run example_project` This will put you in the docker container at a bash shell prompt. You can run `python run.py` which will generate the sample image file in the reports/figures directory.


