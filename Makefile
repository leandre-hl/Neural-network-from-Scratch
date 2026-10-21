CC      = gcc
CFLAGS  = -Wall -Wextra -O2
LDLIBS  = -lm

TARGET  = exe/main.exe

# The project uses a "unity build": main.c #includes the other .c files,
# so only main.c is compiled, but any source or header change must trigger a rebuild.
DEPS    = main.c parameters.h \
          $(wildcard network/*.c network/*.h util/*.c util/*.h MatrixLib/*.c MatrixLib/*.h)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(DEPS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) main.c -o $@ $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)