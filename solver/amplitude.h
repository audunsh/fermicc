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
    void init(electrongas bs, int n_configs, uvec size);

    //internal functions
    u64 to(uint p, uint q, uint r, uint s);  //compressed index
    Col<u64> from(uint i); //expanded index
    ivec intersect1d(ivec A, ivec B);

    //element related functions

    void zeros(); //zero out all elements
    void insert_zeros(); //append a zero element at end of vElements
    uint uiLastind;
    void scan_uvElements();

    void init_amplitudes(); //initialize as amplitude
    void init_t3_amplitudes(); //initialize as amplitude
    void init_interaction(ivec shift);
    void divide_energy(); //divide all elements by corresponding energy (for amplitudes)
    void print_block_maximum();


    void consolidate_blocks(uint uiN, uint tempElementsSize, field<Col<u64> > tempElements, field<uvec> tempBlockmap1,field<uvec> tempBlockmap2,field<uvec> tempBlockmap3);
    void enroll_block(umat umBlock,uint tempElementsSize, uvec tempElements, uvec tempBlockmap1,uvec tempBlockmap2,uvec tempBlockmap3);
    uvec uvNsort1; // = sort_index(uvElements);

    //test
    //uvec uvElemtemp1;// = uvElements.elem(uvNsort);
    Col<u64> uvElemtemp1;

    uvec uvBsort1;// = sort_index(uvNsort); //backsort
    //field<SpMat<uint> > fspBlocks; //sparse blocks with t3 amplitudes
    field<umat> fspBlocks;
    umat fspDims;
    umat getfspBlock(uint i);



    //index related functions
    field<uvec> unpack(uvec vStream, imat imOrder); //unpack a disorganized sequence of indices
    uvec unpack_uvec(uint vStream, imat imOrder);

    //specialized functions
    void map_t3_236_145(ivec Kk_unique);
    void map_t3_623_451(ivec Kk_unique);
    void map_t3_124_356(ivec Kk_unique);
    void map_t3_124_356_new(ivec Kk_unique);




    //external functions
    void map(ivec left, ivec right); //simpler interface to map_regions
    void map_regions(imat L, imat R); //map all regions defined by L == R
    field<uvec> blocksort(ivec LHS, ivec K_unique);

    mat getblock(int u, int i);
    mat getsblock(int u, int i);

    void setblock(int u, int i, mat mBlock);
    void addblock(int u, int i, mat mBlock);



    void addsblock(int u, int i, mat mBlock);



    //Block storage
    int k_step;   //stepsize for identifying unique regions
    vec vElements; //element storage
    vec vElements_temp; //temporary container for permutations and accumulation
    void tempZeros();
    void addblock_temp(int u, int i, mat mBlock);
    mat getsblock_permuted_temp(int u, int i, int n);
    mat getsblock_temp(int u, int i);





    vec vEnergies;
    Col<u64> uvElements; //element storage (prior to initialization)
    //field<umat> fmBlocks; //block of indices

    //field<field <Mat<u64> > > fmBlocks;
    field<field <umat> > fmBlocks;


    field<ivec> fvConfigs; //configuration in quantum numbers of each block

    ivec ivBconfigs; //track aligned configurations

    uvec blocklengths;  //number of blocks in each configuration
    field<imat> fmOrdering; //the ordering of each configuration


    //uvec uvSize; //particle-hole organization
    Col<u64> uvSize; //particle-hole organization


    int iNconfigs;
    uint Np, Nh;
    uint uiCurrent_block;

    field<uvec> permutative_ordering; //for use with unpermuted basic initialization.

    field<uvec> partition_ppp_permutations(field<ivec> LHS, ivec K_unique);
    field<uvec> partition_hhh_permutations(field<ivec> LHS, ivec K_unique);

    field<uvec> partition_pp_permutations(field<ivec> LHS, ivec K_unique);
    field<uvec> partition_hh_permutations(field<ivec> LHS, ivec K_unique);
    field<uvec> Pab;
    field<uvec> Pac;
    field<uvec> Pbc;
    field<uvec> Pij;
    field<uvec> Pik;
    field<uvec> Pjk;



    electrongas eBs;

    //t3 amplitude related
    u64 n0,n1,n2,n3,n4,n5;
    void make_t3();
    u64 to6(u64 p, u64 q, u64 r, u64 s, u64 t, u64 u);
    Col<u64> from6(u64 i);
    void map_regions6(imat L, imat R);
    void map6(ivec left, ivec right);
    void map6c(ivec left, ivec right, ivec preconf);
    void map_regions6c(imat L, imat R, ivec preconf);

    //debugging and optimization
    umat getraw(int u, int i);
    umat getraw_permuted(int u, int i, int n);
    mat getblock_permuted(int u, int i, int n);
    mat getsblock_permuted(int u, int i, int n);
    mat getspblock_permuted(umat aligned, uint i, int n);


    void compress();


    void map_t3_permutations();
    void map_t3_permutations_bconfig();
    void map_t3_permutations_bconfig_sparse();


    void map_t2_permutations();


    //experimental optimized frameworks
    field<ivec> pp();
    field<ivec> ppp(ivec signs);
    field<ivec> ph();
    field<ivec> hp();
    field<ivec> hh();
    field<ivec> hh_compact();
    field<ivec> hhh();

    field<uvec> partition(ivec LHS, ivec K_unique);
    field<uvec> partition_pp(field<ivec> LHS, ivec K_unique);
    field<uvec> partition_ppp(field<ivec> LHS, ivec K_unique);

    field<ivec> hpp();
    field<ivec> php();
    field<ivec> pph();

    field<ivec> phh();
    field<ivec> hph();
    field<ivec> hhp();


    field<uvec> partition_hpp(field<ivec> LHS, ivec K_unique);
    field<uvec> partition_php(field<ivec> LHS, ivec K_unique);
    field<uvec> partition_pph(field<ivec> LHS, ivec K_unique);


    field<uvec> partition_phh(field<ivec> LHS, ivec K_unique);
    field<uvec> partition_hph(field<ivec> LHS, ivec K_unique);
    field<uvec> partition_hhp(field<ivec> LHS, ivec K_unique);



    uint nthreads;
    uint memsize;
    uint debug_enroll;


};

#endif // AMPLITUDE_H
