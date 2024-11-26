CC = gcc
CFLAGS = -Wall -Wextra -I.

SRC_DIR = kernel/core
BUILD_DIR = build
TEST_DIR = test

.PHONY: all clean test

all: $(BUILD_DIR)/kernel

$(BUILD_DIR)/kernel: $(SRC_DIR)/process/process_manager.c $(SRC_DIR)/memory/memory_manager.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $@

test: test_process test_memory

test_process: $(TEST_DIR)/kernel/test_process_manager.c $(SRC_DIR)/process/process_manager.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $(BUILD_DIR)/test_process

test_memory: $(TEST_DIR)/kernel/test_memory_manager.c $(SRC_DIR)/memory/memory_manager.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $(BUILD_DIR)/test_memory

clean:
	rm -rf $(BUILD_DIR)