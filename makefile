sm_version=86
NVCC=/usr/local/cuda-12.4/bin/nvcc
INCLUDES=
OPTIMIZATION=-O0
LINKS=-lcudart -lcuda
OUTPUT=run
KERNEL=main.cu
COMMENT=update

all:
	make kernel

kernel:
	${NVCC} -arch=sm_${sm_version} ${OPTIMIZATION} ${INCLUDES} ${LINKS} -o ${OUTPUT} ${KERNEL}

push:
	git add .
	git commit -m "${COMMENT}"
	git push

run:
	./${OUTPUT}

clean:
	rm -f ${OUTPUT}