#include "blockmat.h"

blockmat::blockmat()
{
}

void blockmat::set_size(uint N){
    p.set_size(N);
    q.set_size(N);
    r.set_size(N);
    s.set_size(N);
    selfSize = N;
}

void blockmat::set_block(uint n, uvec pb, uvec qb, uvec rb, uvec sb){
    p(n) = pb;
    q(n) = qb;
    r(n) = rb;
    s(n) = sb;
}

mat blockmat::get_block(uint n){

}
