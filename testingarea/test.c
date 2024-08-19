#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv){
  (void) argc;
  (void) argv;
  int* numbers = malloc(sizeof(int) * 3);
  char buffer[] = "0,1,2,";
  printf("%s\n", buffer);
  char* token = strtok(buffer, ",");
  for(size_t i = 0; i < 3; ++i){
    printf("%s\n", token);
    token = strtok(NULL, ",");
  }
  

  free(numbers);
  return 0;
}