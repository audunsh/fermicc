#ifndef BCCD_H
#define BCCD_H
#define ARMA_64BIT_WORD
#include <armadillo>

#include "basis/electrongas.h"
#include "solver/amplitude.h"
#include "solver/blockmap.h"

#include <time.h>

using namespace std;
using namespace arma;


// ##################################################
// ##                                              ##
// ## Block fragmented CCD                         ##
// ##                                              ##
// ##################################################

class bccd
{
public:
    bccd(electrongas fgas);

    //internal functions
    double energy(); //calculate energy
    void init(); //initialize all amplitudes and interactions
    void advance(); //perform one advancement of the solution
    void solve(); //iterate until solution converges
    void compare();
    bool unconverged(); //convergence test - returns true while unconverged
    umat intersect_blocks(amplitude a, uint na, blockmap b, uint nb);




    //internal objects and parameters
    electrongas eBs;
    uint Np;
    uint Nh;

    //amplitudes and interactions

    amplitude t2;
    blockmap vhhpp;
    blockmap vpphh;
    blockmap v0;

};

#endif // BCCD_H
