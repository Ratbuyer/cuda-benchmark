sm_version=90a
NVCC=nvcc
INCLUDES=
OPTIMIZATION=-O3
# DEBUG=-DNDEBUG
LINKS=-lcudart -lcuda
OUTPUT=out
KERNEL=1
COMMENT=update

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
	test.cu

push:
	git add .
	git commit -m "${COMMENT}"
	git push

run:
	./${OUTPUT}

clean:
	rm -f ${OUTPUT}