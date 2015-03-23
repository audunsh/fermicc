#ifndef CCD_H
#define CCD_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>
#include "solver/flexmat.h"
#include "solver/initializer.h"
#include "basis/electrongas.h"

using namespace std;
using namespace arma;


class ccd
{
public:
    ccd(electrongas bs);


    electrongas ebs;
};

#endif // CCD_H
