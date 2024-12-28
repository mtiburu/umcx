CXXFLAGS += -g -Wall -pedantic -O3 -std=c++11
BINARY = umcx
SRC = umcx.cpp
OBJS = $(SRC:.cpp=.o)

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
