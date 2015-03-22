#ifndef FLEXMAT_H
#define FLEXMAT_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>

using namespace std;
using namespace arma;

class flexmat
{
public:
    //A class for flexible pphh-matrix manipulations
    flexmat(sp_mat mV, int Np, int Nh);
    int iNp;
    int iNh;

    sp_mat smV;

    sp_mat ab_ij();
    sp_mat ai_bj();




};

#endif // FLEXMAT_H
