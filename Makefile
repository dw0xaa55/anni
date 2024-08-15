CC=gcc
IN=anni.c
OUT=anni
CFLAGS=-Wall -Wextra -lm -g

out:
	$(CC) $(IN) $(CFLAGS) -o $(OUT)

run:
	./$(OUT)
