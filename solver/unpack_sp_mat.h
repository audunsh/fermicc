#ifndef UNPACK_SP_MAT_H
#define UNPACK_SP_MAT_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>

using namespace std;
using namespace arma;

class unpack_sp_mat
{
public:
    unpack_sp_mat(sp_mat c);
    vec vT0;
    vec vT1;
    vec vVals;

};

#endif // UNPACK_SP_MAT_H
