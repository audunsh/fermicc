#ifndef FLEXMAT6_H
#define FLEXMAT6_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>
#include "solver/unpack_sp_mat.h"

using namespace std;
using namespace arma;


class flexmat6
{
public:
    flexmat6();
    int iNp, iNq, iNr, iNs, iNt, iNu;

    void init(vec values, uvec p, uvec q, uvec r, uvec s, uvec t, uvec u, int Np, int Nq, int Nr, int Ns, int Nt, int Nu);

    void map_indices();
    field<uvec> row_indices;
    field<uvec> col_indices;
    field<uvec> col_uniques;
    uvec row_lengths;
    uvec col_lengths;
    uvec cols_i;
    uvec rows_i;
    ivec col_ptrs;
    ivec MCols; //mapping for the columns of the dense block
    ivec all_columns;


    void shed_zeros();

    void set_amplitudes(vec Energy);

    void partition(field<vec> fBlocks);

    void update(sp_mat spC, int Np, int Nq, int Nr, int Ns, int Nt, int Nu);
    vec vEnergy;

    //electrongas eBs;


    vec vValues;
    uvec vp;
    uvec vq;
    uvec vr;
    uvec vs;
    uvec vt;
    uvec vu;

    sp_mat smV;
    sp_mat rows(uvec urows); //returns a identically sized sp_mat with only urows set to non-zero
    mat rows_dense(uvec urows); //returns a identically sized sp_mat with only urows set to non-zero

    umat locations;

    void deinit();

};

#endif // FLEXMAT6_H
