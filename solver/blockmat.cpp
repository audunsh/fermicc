#include "blockmat.h"

blockmat::blockmat()
{
}

void blockmat::set_size(uint N, uint Np, uint Nq, uint Nr, uint Ns){
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
    p(n) = pb;
    q(n) = qb;
    r(n) = rb;
    s(n) = sb;
    //cout << pb << " "<< qb << " "<< rb << " "<< sb << " " << endl;
    //p(n).print();
    uvec req = conv_to<uvec>::from(rb + uNr*sb);
    //cout << req << endl;
    //cout << sb.size() << endl;
    requests(n) = req;
    //requests(n).set_size(sb.size());
    //if(sb.size() == 1){requests(n)(0) = req(0);
    //}
    //else{
    //    requests(n) = req; //when multiplying; which columns to request in matrix to the right
    //}
}

mat blockmat::get_block(uint n){

}
