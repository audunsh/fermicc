#define ARMA_64BIT_WORD
#include <armadillo>

#include <fstream>
#include <iomanip>
#include <time.h>
#include <string>
#include "basis/electrongas.h"
#include "solver/ccsolve.h"




using namespace std;
using namespace arma;

int main()
{
    electrongas fgas;
    fgas.generate_state_list(4,1.0);
    ccsolve solver(fgas);
    return 0;

}

