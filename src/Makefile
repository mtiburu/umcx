ROOTDIR ?= ..
MCXDIR  ?= $(ROOTDIR)
OUTPUT_DIR=$(MCXDIR)/bin
MKDIR = mkdir

CXX ?= clang++

CXXFLAGS += -g -Wall -Wextra -pedantic -std=c++11 -O3 -ffast-math -funroll-loops 

CXXFLAGS += -I${HOME}/sleef/include/ 
LDFLAGS += -L${HOME}/sleef/lib64 


BINARY = umcx
SRC = umcx.cpp
OBJS = $(SRC:.cpp=.o)

OMPACC=-fopenmp $(CPPFLAGS) $(LDFLAGS)
MPICXX=mpicxx 

ifneq (,$(filter $(MAKECMDGOALS),nvc))
    OMPACC=-mp=gpu
endif

ifneq (,$(filter $(ACC),on))
    ifneq (,$(filter $(MAKECMDGOALS),nvc))
        OMPACC=-acc=gpu
    else
        OMPACC=-fopenacc
    endif
endif

nvc: CXX=nvc++
nvc: CXXFLAGS+=-Minfo=mp,accel -Minline -gpu=mem:managed,ptxinfo -static-nvidia
amd: CXXFLAGS+=-foffload=amdgcn-amdhsa="-march=gfx906" -foffload="-lm" -fcf-protection=none -fno-stack-protector
amdclang: CXXFLAGS+=-v -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
    -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 -fcf-protection=none -fno-stack-protector
amdclang: USERLINKFLAGS += -lomp -lm
nvidiaclang: CXX=clang++
nvidiaclang: CXXFLAGS+=-fopenmp-targets=nvptx64 --cuda-gpu-arch=sm_70
nvidia: CXXFLAGS+=-foffload=nvptx-none -foffload="-lm" -fcf-protection=none -fno-stack-protector
nvidiaclang nvidia amd nvc: CXXFLAGS+=-DGPU_OFFLOAD
nvidiaclang nvidia amd nvc: omp

omp: CXXFLAGS+=$(OMPACC)
debugmulti multi: omp
debugmulti multi: CXXFLAGS+=-offload=disable
debugsingle debugmulti: CXXFLAGS+=-DDEBUG
debugsingle debugmulti multi single omp: $(OUTPUT_DIR)/$(BINARY)


sleef: CXXFLAGS+=-I$(HOME)/sleef/include
sleef: LDFLAGS+=-L$(HOME)/sleef/lib64 -lsleef -lm
sleef: omp



# MPI Build
mpi: CXX=$(MPICXX)
mpi: USERLINKFLAGS+=-lmpi	
mpi: CXXFLAGS+=-DMPI_ENABLED -fopenmp
mpi: $(OUTPUT_DIR)/$(BINARY)
 
all: multi

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(OBJS)
	$(CXX)  $(CXXFLAGS) -o $(OUTPUT_DIR)/$(BINARY) $(OBJS) $(USERLINKFLAGS)

%.o: %.c
	$(CC)  $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX)  $(CXXFLAGS) $(USERCXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OUTPUT_DIR)/$(BINARY) *.o

# derived the astyle settings from https://github.com/nlohmann/json
pretty:
	astyle \
	    --style=attach \
	    --indent=spaces=4 \
	    --indent-modifiers \
	    --indent-switches \
	    --indent-preproc-block \
	    --indent-preproc-define \
	    --indent-col1-comments \
	    --pad-oper \
	    --pad-header \
	    --align-pointer=type \
	    --align-reference=type \
	    --add-brackets \
	    --convert-tabs \
	    --close-templates \
	    --lineend=linux \
	    --preserve-date \
	    --suffix=none \
	    --formatted \
	    --break-blocks \
	   "*.cpp"

.PHONY: amdclang
amdclang: $(OUTPUT_DIR)/$(BINARY)
.DEFAULT_GOAL=all
