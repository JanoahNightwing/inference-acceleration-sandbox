CC = gcc
CFLAGS = -Wall -Wextra -O3
SRC = speculative_decoding.c

speculative_decoding: $(SRC) run.c
	$(CC) $(CFLAGS) -o speculative_decoding $(SRC) -lm

debug: $(SRC) run.c
	$(CC) -Wall -Wextra -g -o speculative_decoding $(SRC) -lm

clean:
	rm -f speculative_decoding
