.PHONY: clean

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

SRCS=$(wildcard *.c)
CC=${CLINKER}
PROGS=${SRCS:.c=.out}
OBJS=$(patsubst %.c,%.o,$(SRCS))

all: clean $(PROGS)

%.out: %.o chkopts
	$(CC) -o $@ $< ${PETSC_LIB}
	${RM} *.o

clean::
	rm -rf *.o *.out
