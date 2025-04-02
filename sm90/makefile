sm_version=90a
NVCC=/usr/local/cuda-12.4/bin/nvcc
INCLUDES=
OPTIMIZATION=-O3
# DEBUG=-DNDEBUG
LINKS=-lcudart -lcuda
OUTPUT=out
KERNEL=1
COMMENT=update
BLOCK_SIZE=1
DATA_SIZE=28

all:
	make kernel

kernel:
	${NVCC} -arch=sm_${sm_version} \
	${OPTIMIZATION} \
	${DEBUG} \
	${INCLUDES} \
	${LINKS} \
	-o ${OUTPUT} \
	-DKERNEL=${KERNEL} \
	-DBLOCK_SIZE=${BLOCK_SIZE} \
	-DDATA_SIZE=${DATA_SIZE} \
	main.cu

push:
	git add .
	git commit -m "${COMMENT}"
	git push

run:
	./${OUTPUT}

clean:
	rm -f ${OUTPUT}