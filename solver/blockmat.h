#ifndef BLOCKMAT_H
#define BLOCKMAT_H
#define ARMA_64BIT_WORD
#include <armadillo>

using namespace std;
using namespace arma;

class blockmat
{
public:
    /*
     * A class for storing indices of a partitioned matrix in blocks
     */
    blockmat();
    void set_size(uint N, uint Np, uint Nq, uint Nr, uint Ns);
    void set_block(uint n, uvec pb, uvec qb, uvec rb, uvec sb);
    umat get_sparse_block(uint n);
    field<uvec> get_block(uint n);


    field<uvec> p;
    field<uvec> q;
    field<uvec> r;
    field<uvec> s;
    field<uvec> indices;
    field<uvec> requests;
    u32 uN;
    uint uNp, uNq, uNr, uNs;
    mat locations;
};

#endif // BLOCKMAT_H
