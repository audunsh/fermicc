#include "amplitude.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "basis/electrongas.h"

using namespace std;
using namespace arma;


amplitude::amplitude(electrongas bs, int n_configs)
{
    eBs = bs;
    k_step = 2*eBs.vKx.max()+3; //stepsize for identifying unique regions
    iNconfigs = n_configs;
    fmBlocks.set_size(iNconfigs); //block of indices
    fvConfigs.set_size(iNconfigs); //configuration in quantum numbers of each block
    blocklengths.set_size(iNconfigs);  //number of blocks in each configuration
    fmOrdering.set_size(iNconfigs); //the ordering of each configuration

    Np = eBs.iNbstates-eBs.iNparticles; //conflicting naming here
    Nh = eBs.iNparticles;

    uvSize.set_size(4); //particle-hole organization
    uvSize(0) = Np; //rows
    uvSize(1) = Np;
    uvSize(2) = Nh; //columns
    uvSize(3) = Nh;
}

// ##################################################
// ##                                              ##
// ## Element related functions                    ##
// ##                                              ##
// ##################################################

void amplitude::zeros(){} //zero out all elements
void amplitude::init_amplitudes(){} //initialize as amplitude
void amplitude::divide_energy(){} //divide all elements by corresponding energy (for amplitudes)

// ##################################################
// ##                                              ##
// ## Index related functions                      ##
// ##                                              ##
// ##################################################

uvec amplitude::unpack(vec vStream, imat imOrder, umat umDimension){} //unpack a disorganized sequence of indices

int amplitude::to(int p, int q, int r, int s){
    return p + q*uvSize(0) + r*uvSize(0)*uvSize(1) + s * uvSize(0)*uvSize(1)*uvSize(2);
}  //compressed index

ivec amplitude::from(int i){
    ivec ret(4);
    ret(3) = floor(i/(uvSize(0)*uvSize(1)*uvSize(2)));
    ret(2) = floor((i-ret(3)*uvSize(0)*uvSize(1)*uvSize(2))/(uvSize(0)*uvSize(1)));
    ret(1) = floor((i-ret(3)*uvSize(0)*uvSize(1)*uvSize(2) - ret(2)*uvSize(0)*uvSize(1))/uvSize(0));
    ret(0) = i-ret(3)*uvSize(0)*uvSize(1)*uvSize(2) - ret(2)*uvSize(0)*uvSize(1) - ret(1)*uvSize(0);
    return ret;
} //expanded index

// ##################################################
// ##                                              ##
// ## External functions                           ##
// ##                                              ##
// ##################################################

void amplitude::map_regions(imat L, imat R){} //map all regions defined by L == R
ivec amplitude::match_config(int u, ivec ivConfig){} //retrieve all
mat amplitude::getblock(int u, int i){}
mat amplitude::setblock(int u, int i, mat mBlock){}
mat amplitude::addblock(int u, int i, mat mBlock){}
