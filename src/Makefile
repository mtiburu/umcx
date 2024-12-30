CXXFLAGS += -g -Wall -Wextra -std=c++14 -O3 #-pedantic
BINARY = umcx
SRC = umcx.cpp
OBJS = $(SRC:.cpp=.o)

nvc: CXX=nvc++
nvc: CXXFLAGS+=-mp=gpu -Minfo=mp,accel -Minline -gpu=mem:managed -DGPU_OFFLOAD
amd: CXXFLAGS+=-foffload=amdgcn-amdhsa="-march=gfx906" -fcf-protection=none -fno-stack-protector -DGPU_OFFLOAD
nvidia: CXXFLAGS+=-foffload=nvptx-none -fcf-protection=none -fno-stack-protector -DGPU_OFFLOAD
nvidia amd nvc: multi

debugmulti multi: CXXFLAGS+=-fopenmp
debugsingle debugmulti: CXXFLAGS+=-DDEBUG
debugsingle debugmulti multi single: $(BINARY)

all: multi

$(BINARY): $(OBJS)
	$(CXX)  $(CXXFLAGS) -o $(BINARY) $(OBJS)

%.o: %.c
	$(CC)  $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX)  $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BINARY) *.o

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
