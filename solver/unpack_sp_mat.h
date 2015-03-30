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
    uvec vT0;
    uvec vT1;
    vec vVals;
    mat mLocations;

};

#endif // UNPACK_SP_MAT_H
