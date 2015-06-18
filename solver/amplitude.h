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
    amplitude();
    amplitude(electrongas bs, int n_configs, uvec size);

    //internal functions
    uint to(uint p, uint q, uint r, uint s);  //compressed index
    uvec from(uint i); //expanded index
    ivec intersect1d(ivec A, ivec B);

    //element related functions

    void zeros(); //zero out all elements
    void init_amplitudes(); //initialize as amplitude
    void init_interaction(ivec shift);
    void divide_energy(); //divide all elements by corresponding energy (for amplitudes)
    void print_block_maximum();

    //index related functions
    field<uvec> unpack(uvec vStream, imat imOrder); //unpack a disorganized sequence of indices
    uvec unpack_uvec(uint vStream, imat imOrder);


    //external functions
    void map(ivec left, ivec right); //simpler interface to map_regions
    void map_regions(imat L, imat R); //map all regions defined by L == R
    ivec match_config(int u, ivec ivConfig); //retrieve all
    mat getblock(int u, int i);
    void setblock(int u, int i, mat mBlock);
    void addblock(int u, int i, mat mBlock);

    //Block storage
    int k_step;   //stepsize for identifying unique regions
    vec vElements; //element storage
    vec vEnergies;
    uvec uvElements; //element storage (prior to initialization)
    //field<umat> fmBlocks; //block of indices
    field<field <umat> > fmBlocks;
    field<ivec> fvConfigs; //configuration in quantum numbers of each block
    uvec blocklengths;  //number of blocks in each configuration
    field<imat> fmOrdering; //the ordering of each configuration
    uvec uvSize; //particle-hole organization
    int iNconfigs;
    int Np, Nh;
    uint uiCurrent_block;


    electrongas eBs;








};

#endif // AMPLITUDE_H
