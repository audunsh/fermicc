#include "bccd.h"
#define ARMA_64BIT_WORD
#include <armadillo>

#include "basis/electrongas.h"
#include "solver/amplitude.h"
#include <time.h>

using namespace std;
using namespace arma;


bccd::bccd(electrongas fgas)
{
    eBs = fgas;
    Np = fgas.iNbstates-fgas.iNparticles;
    Nh = fgas.iNparticles; //conflicting naming here
    init();
    cout << "Energy:" << energy() << endl;
}

void bccd::init(){
    amplitude tt2(eBs, 3, {Np, Np, Nh, Nh});
    amplitude tvhhpp(eBs, 3, {Nh,Nh,Np,Np});
    blockmap vv(eBs, 3, {Nh,Nh,Np,Np});
    v0 = vv;
    v0.init_interaction({0,0,Nh,Nh});
    t2 = tt2;
    vhhpp = tvhhpp;
    t2.map({1,2}, {3,4});
    //vhhpp.map({1,2}, {3,4});
    t2.init_amplitudes();
    vhhpp.init_interaction({0,0,Nh,Nh});
    t2.divide_energy();

}

umat bccd::intersect_blocks(amplitude a, uint na, amplitude b, uint nb){
    // ############################################
    // ## Find corresponding blocks in a and b   ##
    // ############################################
    umat tintersection(a.blocklengths(na), 2);
    uint counter = 0;
    for(uint n1 = 0; n1 < a.blocklengths(na); ++n1){
        int ac = a.fvConfigs(na)(n1);
        for(uint n2 = 0; n2 < b.blocklengths(nb); ++n2){
            if(b.fvConfigs(nb)(n2) == ac){
                //found intersecting configuration
                tintersection(counter, 0) = n1;
                tintersection(counter, 1) = n2;
                counter +=1;
            }
        }
    }
    //Flatten intersection
    umat intersection(counter-1, 2);
    for(uint n = 0; n < counter; ++n){
        intersection(n, 0) = tintersection(n,0);
        intersection(n, 1) = tintersection(n,1);
    }
    return intersection;
}

double bccd::energy(){
    //uint n = t2.blocklengths(0);
    double e = 0;
    umat c = intersect_blocks(t2,0,vhhpp,0); //this should be calculated prior to function calls (efficiency)
    for(uint i = 0; i < c.n_rows; ++i){
        mat block = vhhpp.getblock(0, c(i,1))*t2.getblock(0,c(i,0));
        cout << "--------------" << endl;
        vhhpp.getblock(0, c(i,1)).print();
        cout << endl;
        v0.getblock(0, c(i,1)).print();


        vec en = block.diag();
        for(uint j = 0; j< en.n_rows; ++j){
            e += en(j);
        }
    }
    return .25*e;
}
