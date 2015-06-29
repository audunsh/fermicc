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
    void init(electrongas bs, int n_configs, uvec size);

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
    field<uvec> blocksort(ivec LHS, ivec K_unique);
    field<uvec> blocksort_symmetric(ivec K_unique);


    //external functions
    void map(ivec left, ivec right); //simpler interface to map_regions
    void map_regions(imat L, imat R); //map all regions defined by L == R
    void map_vpppp();

    void map_vppph();

    mat getblock(int u, int i);
    mat getblock_vpppp(int u, int i);


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
    uint Np, Nh;
    uint uiCurrent_block;


    electrongas eBs;



    field<ivec> pp();
    field<ivec> ppp();
    field<ivec> ph();
    field<ivec> hp();
    field<ivec> hh();

    field<uvec> partition(ivec LHS, ivec K_unique);
    field<uvec> partition_pp(field<ivec> LHS, ivec K_unique);
    field<uvec> partition_ppp(field<ivec> LHS, ivec K_unique);




};

#endif // BLOCKMAP_H
