EXECUTABLE := cuda_parser
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
CU_FILES   := cuda_parser_thd.cu
CU_DEPS    :=
CC_FILES   := main.cpp read_grammar.cpp
LOGS	   := logs

all: $(EXECUTABLE)

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -g -std=c++11
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 -std=c++11 --gpu-architecture compute_61
LIBS += GL glut cudart

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/read_grammar.o $(OBJDIR)/cuda_parser_thd.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) *.ppm


$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)




$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
