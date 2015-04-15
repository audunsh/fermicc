#ifndef BLOCKMAT_H
#define BLOCKMAT_H
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
    void set_size(uint N);
    void set_block(uint n, uvec pb, uvec qb, uvec rb, uvec sb);
    mat get_block(uint n);


    field<uvec> p;
    field<uvec> q;
    field<uvec> r;
    field<uvec> s;
    u32 selfSize;

};

#endif // BLOCKMAT_H
