#ifndef BLOCKMAP_H
#define BLOCKMAP_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include "basis/electrongas.h"

using namespace std;
using namespace arma;

class blockmap
{
public:
    blockmap();
    blockmap(electrongas bs, int n_configs, uvec size);

    //internal functions
    uint to(uint p, uint q, uint r, uint s);  //compressed index
    uvec from(uint i); //expanded index
    ivec intersect1d(ivec A, ivec B);

    //element related functions
    void init_interaction(ivec shift);


    void print_block_maximum();

    //index related functions
    field<uvec> unpack(uvec vStream, imat imOrder); //unpack a disorganized sequence of indices
    uvec unpack_uvec(uint vStream, imat imOrder);


    //external functions
    void map(ivec left, ivec right); //simpler interface to map_regions
    void map_regions(imat L, imat R); //map all regions defined by L == R
    ivec match_config(int u, ivec ivConfig); //retrieve all
    mat getblock(int u, int i);
    mat setblock(int u, int i, mat mBlock);
    mat addblock(int u, int i, mat mBlock);

    //Block storage
    int k_step;   //stepsize for identifying unique regions
    vec vElements; //element storage
    vec vEnergies;
    uvec uvElements; //element storage (prior to initialization)
    //field<umat> fmBlocks; //block of indices
    field<field <umat> > fmBlocks;
    field<field <uvec> > fmBlockz;

    field<field <uvec> > fuvRows;
    field<field <uvec> > fuvCols;

    field<ivec> fvConfigs; //configuration in quantum numbers of each block
    uvec blocklengths;  //number of blocks in each configuration
    field<imat> fmOrdering; //the ordering of each configuration
    uvec uvSize; //particle-hole organization
    int iNconfigs;
    int Np, Nh;
    uint uiCurrent_block;


    electrongas eBs;



};

#endif // BLOCKMAP_H