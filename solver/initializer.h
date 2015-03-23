#ifndef INITIALIZER_H
#define INITIALIZER_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>
#include "solver/flexmat.h"
#include "basis/electrongas.h"

using namespace std;
using namespace arma;

class initializer
{
public:
    initializer();
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
    //flexmat fmVpphh;
    SpMat<double> Vhhhh;
    SpMat<double> Vhpph;

    uvec aVpppp;
    uvec bVpppp;
    uvec cVpppp;
    uvec dVpppp;
    vec vValsVpppp;

    uvec iVhhhh;
    uvec jVhhhh;
    uvec kVhhhh;
    uvec lVhhhh;
    vec vValsVhhhh;

    uvec aVpphh;
    uvec bVpphh;
    uvec iVpphh;
    uvec jVpphh;
    vec vValsVpphh;

    uvec iVhhpp;
    uvec jVhhpp;
    uvec aVhhpp;
    uvec bVhhpp;
    vec vValsVhhpp;

    uvec iVhpph;
    uvec aVhpph;
    uvec bVhpph;
    uvec jVhpph;
    vec vValsVhpph;

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
