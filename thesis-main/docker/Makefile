YML=docker-compose.yml
GPU=0 # Set GPU number [int e.g. 0, 1, 2 .. or all]

# Get port number from hostname by cutting the first 2 characters (dl13 -> 13)
HOSTNUMBER=$(shell hostname | cut -c 3-)
P_FO=$(HOSTNUMBER)051
P_FO_2=$(HOSTNUMBER)052
P_TF=$(HOSTNUMBER)066

all:
	echo "Usage:"
	echo "	make image     		# build image"
	echo "	make up        		# start container"
	echo "	make exec      		# go inside container"
	echo "	make down      		# stop container"
	echo "  make mount     		# mount diskstation caviar folder"
	echo "  make check_mount 	# print mounted diskstation folders"

vars:
	echo "$(HOSTNUMBER)"
	echo "GPU=" $(GPU)
	echo "P_FO=" $(P_FO)
	echo "P_TF=" $(P_TF)

image:
	env GPU=$(GPU) PORT_FIFTYONE=$(P_FO) PORT_FIFTYONE_TWO=$(P_FO_2) PORT_TFBOARD=$(P_TF) docker-compose -f $(YML) build

up:
	env GPU=$(GPU) PORT_FIFTYONE=$(P_FO) PORT_FIFTYONE_TWO=$(P_FO_2) PORT_TFBOARD=$(P_TF) docker-compose -f $(YML) up -d
	docker exec thesis bash docker/init.sh

attach:
	docker attach thesis

exec:
	docker exec -it thesis bash

down:
	env GPU=$(GPU) PORT_FIFTYONE=$(P_FO) PORT_FIFTYONE_TWO=$(P_FO_2) PORT_TFBOARD=$(P_TF) docker-compose -f $(YML) down

mount:
	mkdir -p /data/home/$(USER)/diskstation/caviar
	mkdir -p /data/home/$(USER)/data
	smbmount diskstationii1 caviar/security/datasets /data/home/$(USER)/diskstation/caviar
	smbmount diskstationii1 data/corselcw /data/home/$(USER)/data

check_mount:
	mount | grep -w cifs

