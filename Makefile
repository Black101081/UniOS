CC = gcc
CFLAGS = -Wall -Wextra -I./kernel -I./api

KERNEL_SRC = kernel/file_manager.c kernel/memory_manager.c kernel/process_manager.c kernel/sandbox_manager.c
API_SRC = api/api_core.c
UI_SRC = ui/ui_main.c
TEST_SRC = test/test_file_manager.c

all: ui_main test_file_manager

ui_main: $(KERNEL_SRC) $(API_SRC) $(UI_SRC)
	$(CC) $(CFLAGS) -o $@ $^

test_file_manager: $(KERNEL_SRC) $(TEST_SRC)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f ui_main test_file_manager