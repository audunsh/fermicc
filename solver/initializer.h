#ifndef INITIALIZER_H
#define INITIALIZER_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>
#include "basis/electrongas.h"

using namespace std;
using namespace arma;

class initializer
{
public:
    initializer(electrongas Bs);

    vec V(uvec p, uvec q, uvec r, uvec s); //vectorized interactions
    vec V2(uvec t0, uvec t1); //vectorized interactions

    void sVpppp();
    void sVhhhh();
    void sVpphh();
    void sVhhpp();
    void sVhpph();

    //support functions
    uvec append(uvec V1, uvec V2);
    vec appendvec(vec V1, vec V2);
    vec absdiff2(vec kpx, vec kpy, vec kpz, vec kqx,vec kqy, vec kqz);

    electrongas bs;

    SpMat<double> Vpppp;
    SpMat<double> Vhhpp;
    SpMat<double> Vpphh;
    SpMat<double> Vhhhh;
    SpMat<double> Vhpph;

    int iNp;
    int iNh;
    int iNh2;
    int iNp2;
    int iNhp;



    int iNmax;
    int iNmax2;
    int iNmax3;

    double pi = 4*atan(1);


};

#endif // INITIALIZER_H
