ROOTDIR ?= ..
MCXDIR  ?= $(ROOTDIR)
OUTPUT_DIR=$(MCXDIR)/bin
MKDIR = mkdir

CXXFLAGS += -g -Wall -Wextra -pedantic -std=c++11 -O3
BINARY = umcx
SRC = umcx.cpp
OBJS = $(SRC:.cpp=.o)

nvc: CXX=nvc++
nvc: CXXFLAGS+=-mp=gpu -Minfo=mp,accel -Minline -gpu=mem:managed -static-nvidia
amd: CXXFLAGS+=-foffload=amdgcn-amdhsa="-march=gfx906" -foffload="-lm" -fcf-protection=none -fno-stack-protector
nvidiaclang: CXX=clang++
nvidiaclang: CXXFLAGS+=-fopenmp-targets=nvptx64 --cuda-gpu-arch=sm_50
nvidia: CXXFLAGS+=-foffload=nvptx-none -foffload="-lm" -fcf-protection=none -fno-stack-protector
nvidiaclang nvidia amd nvc: CXXFLAGS+=-DGPU_OFFLOAD
nvidiaclang nvidia amd nvc: omp

omp: CXXFLAGS+=-fopenmp
debugmulti multi: omp
debugmulti multi: CXXFLAGS+=-foffload=disable
debugsingle debugmulti: CXXFLAGS+=-DDEBUG
debugsingle debugmulti multi single omp: $(OUTPUT_DIR)/$(BINARY)

all: multi

makedirs:
	@if test ! -d $(OUTPUT_DIR); then $(MKDIR) $(OUTPUT_DIR); fi

$(OUTPUT_DIR)/$(BINARY): makedirs $(OBJS)
	$(CXX)  $(CXXFLAGS) -o $(OUTPUT_DIR)/$(BINARY) $(OBJS)

%.o: %.c
	$(CC)  $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX)  $(CXXFLAGS) -c $< -o $@

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

.DEFAULT_GOAL=all
