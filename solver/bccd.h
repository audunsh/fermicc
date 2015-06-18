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
    void solve(uint Nt);
    void compare();
    bool unconverged(); //convergence test - returns true while unconverged
    umat intersect_blocks(amplitude a, uint na, blockmap b, uint nb);


    //internal objects and parameters
    electrongas eBs;
    uint Np;
    uint Nh;

    //amplitudes and interactions

    amplitude t2;
    amplitude t2temp; //used for permutations
    amplitude t2n; //next
    blockmap vhhpp;
    blockmap vpppp;
    blockmap vhpph;
    blockmap vpphh;
    blockmap vhhhh;
    blockmap v0;

};

#endif // BCCD_H
