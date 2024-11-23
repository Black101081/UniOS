CC = gcc
CFLAGS = -Wall -Wextra -I.

SRC_DIR = kernel/core
BUILD_DIR = build
TEST_DIR = test

.PHONY: all clean test

all: $(BUILD_DIR)/kernel

$(BUILD_DIR)/kernel: $(SRC_DIR)/*.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@

test: $(TEST_DIR)/*.c
	$(CC) $(CFLAGS) $^ -o $(BUILD_DIR)/test

clean:
	rm -rf $(BUILD_DIR)
