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
    umat intersect_blocks_triple(amplitude a, uint na, blockmap b, uint nb, amplitude c, uint nc);


    bool pert_triples; //enable perturbative triples
    string mode; //mode of operation ( CCD or CCD(T) )

    //internal objects and parameters
    electrongas eBs;
    uint Np;
    uint Nh;

    //amplitudes and interactions

    amplitude t3;
    amplitude t3temp;

    amplitude t2;
    amplitude t2temp; //used for permutations
    amplitude t2temp2;
    amplitude t2n; //next
    blockmap vhhpp;
    blockmap vpppp;
    blockmap vhpph;
    blockmap vpphh;
    blockmap vhhhh;


    //for the triples
    blockmap vppph;
    blockmap vhphh;

    blockmap vphpp;
    blockmap vhhhp;


    blockmap v0;


    void activate_diagrams();
    uint acL1;
    uint acL2;
    uint acL3;
    uint acL4;

    uint acQ1;
    uint acQ2;
    uint acQ3;
    uint acQ4;

    uint acT2a;
    uint acT2b;

    uint acD10c;
    uint acD10b;



};

#endif // BCCD_H
