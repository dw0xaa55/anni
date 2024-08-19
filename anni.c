/* DESCRIPTION: Artificial Neural Network Intelligence   */
/* AUTHOR     : C. Huffenbach                            */
/* VERSION    : 0.1                                      */
/* DATE       : 08/2024                                  */

/* TODO:                                                 */
/*   - [X] think of trainingset architecture             */
/*   - [X] load trainingsets from file                   */
/*   - [ ] option to print networks                      */
/*   - [ ] save trained network to file                  */
/*   - [ ] option to load networks from file             */
/*   - [ ] create Network struct                         */

#define ANNI_IMPLEMENTATION
#include "anni.h"

int main(int argc, char** argv){
  (void) argc;
  (void) argv;

  Trainingset ts;
  load_training_file(&ts,"trainingfile.ts");
  print_trainingset(ts);
  free_trainingset(ts);
  
  return 0;
}
