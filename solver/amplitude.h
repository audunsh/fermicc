#ifndef AMPLITUDE_H
#define AMPLITUDE_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include "basis/electrongas.h"

using namespace std;
using namespace arma;

class amplitude
{
public:
    amplitude(electrongas bs, int n_configs);

    //internal functions
    int to(int p, int q, int r, int s);  //compressed index
    ivec from(int i); //expanded index

    //element related functions
    void zeros(); //zero out all elements
    void init_amplitudes(); //initialize as amplitude
    void divide_energy(); //divide all elements by corresponding energy (for amplitudes)

    //index related functions
    uvec unpack(vec vStream, imat imOrder, umat umDimension); //unpack a disorganized sequence of indices

    //external functions
    void map_regions(imat L, imat R); //map all regions defined by L == R
    ivec match_config(int u, ivec ivConfig); //retrieve all
    mat getblock(int u, int i);
    mat setblock(int u, int i, mat mBlock);
    mat addblock(int u, int i, mat mBlock);

    //Block storage
    int k_step;   //stepsize for identifying unique regions
    vec vElements; //element storage
    field<umat> fmBlocks; //block of indices
    field<vec> fvConfigs; //configuration in quantum numbers of each block
    uvec blocklengths;  //number of blocks in each configuration
    field<mat> fmOrdering; //the ordering of each configuration
    uvec uvSize; //particle-hole organization
    int iNconfigs;
    int Np, Nh;


    electrongas eBs;








};

#endif // AMPLITUDE_H
