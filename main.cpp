#include <fstream>
#include <iomanip>
#include <time.h>
#include <armadillo>
#include <string>
#include "basis/electrongas.h"
#include "solver/ccsolve.h"



using namespace std;
using namespace arma;

int main()
{
    electrongas fgas;
    fgas.generate_state_list(36,1.0);
    //ccsolve solver(fgas);

}

