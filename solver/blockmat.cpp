#include "blockmat.h"
#define ARMA_64BIT_WORD

using namespace std;
using namespace arma;

blockmat::blockmat()
{
}

void blockmat::set_size(uint N, uint Np, uint Nq, uint Nr, uint Ns){
    //set number of blocks
    p.set_size(N);
    q.set_size(N);
    r.set_size(N);
    s.set_size(N);
    requests.set_size(N);
    uNp = Np; //number of states (particles and holes) for each index
    uNq = Nq;
    uNr = Nr;
    uNs = Ns;
    uN = N;  //number of blocks

}

void blockmat::set_block(uint n, uvec pb, uvec qb, uvec rb, uvec sb){
    //set indices/region in block n
    p(n) = pb;
    q(n) = qb;
    r(n) = rb;
    s(n) = sb;
    uvec req = conv_to<uvec>::from(rb + uNr*sb);
    requests(n) = req;

}

field<uvec> blockmat::get_block(uint n){
    //returns 4xN matrix (N = number of elements) with pqrs indices
    uint nx = p(n).size();
    uint ny = r(n).size();
    field<uvec> indices(5);
    indices(0) = p(n);
    indices(1) = q(n);
    indices(2) = r(n);
    indices(3) = s(n);
    indices(4) = requests(n);
    return indices;
}
