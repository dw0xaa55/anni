/* DESCRIPTION: Artificial Neural Network Intelligence   */
/* AUTHOR     : C. Huffenbach                            */
/* VERSION    : 0.2                                      */
/* DATE       : 08/2024                                  */

/* TODO:                                                 */
/*   - [X] think of trainingset architecture             */
/*   - [X] create Network struct                         */
/*   - [X] load trainingsets from file                   */
/*   - [X] option to print networks                      */
/*   - [ ] save trained network to file                  */
/*   - [ ] option to load networks from file             */


#define ANNI_IMPLEMENTATION
#include "anni.h"

int main(int argc, char** argv){
  (void) argc;
  (void) argv;

  Trainingset ts;
  NeuralNetwork nn;
  size_t topology[3] = { 2,2,1 };

  load_training_file(&ts,"trainingfile.ts");
  print_trainingset(ts);
  initialize_model(&nn, topology, GET_CONTAINER_SIZE(topology));
  print_model(nn);

  free_model(nn);
  free_trainingset(ts);
  
  return 0;
}
