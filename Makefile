IN=anni.c
OUT=anni
CC=gcc
CFLAGS=-Wall -Wextra -O2 -lm -g

out:
	$(CC) $(IN) $(CFLAGS) -o $(OUT)

run:
	./$(OUT)
