#include "initializer.h"
#include "basis/electrongas.h"
#include "solver/flexmat.h"
#include <time.h>
#include <solver/blockmat.h>


#define ARMA_64BIT_WORD
#include <armadillo>

using namespace std;
using namespace arma;

initializer::initializer(){}

initializer::initializer(electrongas Bs)
{
    bs = Bs;
    iNp = bs.iNbstates-bs.iNparticles; //number of particle states
    iNh = bs.iNparticles; //number of hole states
    iNp2 = iNp*iNp;
    iNh2 = iNh*iNh;
    iNhp = iNh*iNp;
    iNmax = 2*bs.vKx.max()+2;
    iNmax2 = iNmax*iNmax;
}



vec initializer::V(uvec p, uvec q, uvec r, uvec s){
    //delta function of summation of momentum quantum numbers is assumed to have passed before entering here
    vec ret; // = zeros(p.size());
    ret.set_size(p.size());

    //retrieve relevant quantum numbers (in vectors)
    ivec Msa = bs.vMs.elem(p);
    ivec Msb = bs.vMs.elem(q);
    ivec Msc = bs.vMs.elem(r);
    ivec Msd = bs.vMs.elem(s);

    ivec Kax = bs.vKx.elem(p);
    ivec Kay = bs.vKy.elem(p);
    ivec Kaz = bs.vKz.elem(p);

    ivec Kbx = bs.vKx.elem(q);
    ivec Kby = bs.vKy.elem(q);
    ivec Kbz = bs.vKz.elem(q);

    ivec Kcx = bs.vKx.elem(r);
    ivec Kcy = bs.vKy.elem(r);
    ivec Kcz = bs.vKz.elem(r);

    ivec Kdx = bs.vKx.elem(s);
    ivec Kdy = bs.vKy.elem(s);
    ivec Kdz = bs.vKz.elem(s);

    //set up interaction
    ivec KDplus = conv_to<ivec>::from(ones(p.size())); //*4*pi/bs.dL3;
    KDplus = KDplus%(Kax+Kbx==Kcx+Kdx)%(Kay+Kby==Kcy+Kdy)%(Kaz+Kbz==Kcz+Kdz);

    ivec diff_ca = absdiff2(Kcx, Kcy, Kcz, Kax, Kay, Kaz);
    ivec diff_da = absdiff2(Kdx, Kdy, Kdz, Kax, Kay, Kaz);

    vec term1 = zeros(p.size()); //term 1
    //term1.set_size(p.size());
    term1.elem(conv_to<uvec>::from(find(Msa==Msc && Msb==Msd))) += 4*pi/bs.dL3; //By changing this i get comparable results
    term1.elem(conv_to<uvec>::from(find(Kax==Kcx && Kay==Kcy && Kaz==Kcz))) *= 0;
    term1.elem(find(term1!=0)) /= 4*pi*pi*conv_to<vec>::from(diff_ca.elem(find(term1!=0)))/bs.dL2;

    vec term2 = zeros(p.size()); //term 2
    //term2.set_size(p.size());
    term2.elem(conv_to<uvec>::from(find(Msa==Msd && Msb==Msc))) += 4*pi/bs.dL3;
    term2.elem(conv_to<uvec>::from(find(Kax==Kdx && Kay==Kdy && Kaz==Kdz))) *= 0;
    term2.elem(find(term2!=0)) /= 4*pi*pi*conv_to<vec>::from(diff_da.elem(find(term2!=0)))/bs.dL2;

    return KDplus%(term1 - term2);

}

vec initializer::V3(uvec p, uvec q, uvec r, uvec s){
    //Inefficient interaction calculation, not really vectorized but returns a vector
    vec vVals;
    vVals.set_size(p.size());
    for(int n = 0; n< p.size(); n++){
        vVals(n) = bs.v2(p(n), q(n), r(n), s(n));
    }
    return vVals;
}

vec initializer::V4(Col<u32> p, Col<u32> q, Col<u32> r, Col<u32> s){
    //Inefficient interaction calculation, not really vectorized but returns a vector
    arma::u32 nnz = p.size();
    double * aux_mem = new double[nnz];
    vec vVals(aux_mem, nnz, false, true);

    for(int n = 0; n< p.size(); n++){
        vVals(n) = bs.v2(p(n), q(n), r(n), s(n));
    }
    return vVals;
}


vec initializer::V2(uvec t0, uvec t1){
    //delta function of summation of momentum quantum numbers is assumed to have passed before entering here
    uvec b = conv_to<uvec>::from(floor(t0/iNp)); //convert to unsigned integer indexing vector
    uvec a = conv_to<uvec>::from(t0) - b*iNp;
    uvec d = conv_to<uvec>::from(floor(t1/iNp)); //convert to unsigned integer indexing vector
    uvec c = conv_to<uvec>::from(t1) - d*iNp;

    vec ret = zeros(t0.size());

    ivec Msa = bs.vMs.elem(a+iNh);
    ivec Msb = bs.vMs.elem(b+iNh);
    ivec Msc = bs.vMs.elem(c+iNh);
    ivec Msd = bs.vMs.elem(d+iNh);

    ivec Kax = bs.vKx.elem(a+iNh);
    ivec Kay = bs.vKy.elem(a+iNh);
    ivec Kaz = bs.vKz.elem(a+iNh);

    ivec Kbx = bs.vKx.elem(b+iNh);
    ivec Kby = bs.vKy.elem(b+iNh);
    ivec Kbz = bs.vKz.elem(b+iNh);

    ivec Kcx = bs.vKx.elem(c+iNh);
    ivec Kcy = bs.vKy.elem(c+iNh);
    ivec Kcz = bs.vKz.elem(c+iNh);

    ivec Kdx = bs.vKx.elem(d+iNh);
    ivec Kdy = bs.vKy.elem(d+iNh);
    ivec Kdz = bs.vKz.elem(d+iNh);


    ivec KDplus = conv_to<ivec>::from(ones(t0.size())); //*4*pi/bs.dL3;
    KDplus = KDplus%(Kax+Kbx==Kcx+Kdx)%(Kay+Kby==Kcy+Kdy)%(Kaz+Kbz==Kcz+Kdz);

    ivec diff_ca = absdiff2(Kcx, Kcy, Kcz, Kax, Kay, Kaz);
    ivec diff_da = absdiff2(Kdx, Kdy, Kdz, Kax, Kay, Kaz);

    vec term1 = zeros(t0.size()); //term 1
    //term1.elem(conv_to<uvec>::from(find(Msa==Msc && Msb==Msd))) += 4*pi/bs.dL3;
    //term1.elem(conv_to<uvec>::from(find(Kax==Kcx && Kay==Kcy && Kaz==Kcz))) *= 0;
    //term1.elem(find(term1!=0)) /= 4*pi*pi*diff_ca.elem(find(term1!=0))/bs.dL2;

    vec term2 = zeros(t0.size()); //term 2
    //term2.elem(conv_to<uvec>::from(find(Msa==Msd && Msb==Msc))) += 4*pi/bs.dL3;
    //term2.elem(conv_to<uvec>::from(find(Kax==Kdx && Kay==Kdy && Kaz==Kdz))) *= 0;
    //term2.elem(find(term2!=0)) /= 4*pi*pi*diff_da.elem(find(term2!=0))/bs.dL2;

    return KDplus%(term1 - term2);

}

ivec initializer::absdiff2(ivec kpx, ivec kpy, ivec kpz, ivec kqx,ivec kqy, ivec kqz){
    //vectorized absdiff2 |kp - kq|^2
    ivec ret1 = kpx-kqx;
    ivec ret2 = kpy-kqy;
    ivec ret3 = kpz-kqz;
    return ret1%ret1 + ret2%ret2 + ret3%ret3;

}

uvec initializer::append(uvec V1, uvec V2){
    //
    int V1size = V1.size();
    V1.resize(V1size+V2.size());
    for(int i= 0; i<V2.size(); i++){
        V1(i+V1size) = V2(i);
    }
    return V1;
}


vec initializer::appendvec(vec V1, vec V2){
    //int V1size = V1.size();
    vec V3;
    V3.set_size(V1.size() + V2.size());
    for(int i= 0; i<V1.size(); i++){
        V3(i) = V1(i);
    }
    for(int i= V1.size(); i<V1.size()+V2.size(); i++){
        V3(i) = V2(i-V1.size());
    }

    return V3;
}


void initializer::sVppppO(){
    //optimized interaction for Vpppp
    clock_t  t;
    t = clock();
    //using symmetries, only consider a>b>:
    uvec A, B; //vectors containing indices (row and column)
    B.set_size(iNp*((iNp+1.0)/2.0));
    A.set_size(iNp*((iNp+1.0)/2.0));

    uint n = 0;
    for(uint a = 0; a<iNp; ++a){
        for(uint b = a; b<iNp; ++b){
            A(n) = a;
            B(n) = b;
            n += 1;
        }
    }




    //careful here
    //vec AB = linspace(0,iNp2-1,iNp2);
    //uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    //uvec A = conv_to<uvec>::from(AB) - B*iNp;


    cout << "Good so far... (1)"  << endl;




    cout << "Good so far... (2)"  << (double)(clock() - t)/CLOCKS_PER_SEC<< endl;
    t = clock();

    //Setting up a vector containint a unique integer identifier for K + M_s
    ivec KABx = bs.vKx.elem(A+iNh)+bs.vKx.elem(B+iNh);
    ivec KABy = iNmax*(bs.vKy.elem(A+iNh)+bs.vKy.elem(B+iNh));
    ivec KABz = iNmax2*(bs.vKz.elem(A+iNh)+bs.vKz.elem(B+iNh));
    ivec KABms = iNmax*iNmax2*(bs.vMs(A+iNh) + bs.vMs(B + iNh));

    ivec KAB = KABx+KABy+KABz + KABms;
    ivec KAB_unique = unique(KAB);

    field<uvec> TT;
    TT.set_size(KAB_unique.size(), 2);
    u32 iN = 0;
    vec T, O;
    uvec tT;
    uvec t0, t1;

    cout << KAB_unique.size() << endl;

    bmVpppp.set_size(KAB_unique.size(), iNp, iNp, iNp, iNp);

    cout << "Good so far... (3)"  << (double)(clock() - t)/CLOCKS_PER_SEC<< endl;
    t = clock();



    for(uint i = 0; i < KAB_unique.size(); ++i){
        //locating non-zero regions where K_a + K_b = K_c + K_d
        //it is possible to exploit spin symmetry further inside this loop
        T = conv_to<vec>::from(find(KAB==KAB_unique(i))); //Is it possible to make this vector "shrink" as more indices is identified?

        tT = find(KAB==KAB_unique(i));
        bmVpppp.set_block(i, A.elem(tT), B.elem(tT),A.elem(tT), B.elem(tT));

        O = ones(T.size());
        t0 = conv_to<uvec>::from(kron(T, O));
        t1 = conv_to<uvec>::from(kron(O, T));
        TT(i, 0) = t0;
        TT(i, 1) = t1;
        iN += t0.size();
    }




    cout << "Good so far... (4)"  << (double)(clock() - t)/CLOCKS_PER_SEC<< endl;
    t = clock();

    uvec aVppp(iN);
    uvec bVppp(iN);
    uvec cVppp(iN);
    uvec dVppp(iN);

    cout << "Good so far... (5)"  << (double)(clock() - t)/CLOCKS_PER_SEC<< endl;
    t = clock();

    iN = 0;
    int iNt = 0;
    for(uint i = 0; i < KAB_unique.size(); ++i){
        aVppp(span(iN, iN+TT(i,0).size()-1)) = A.elem(TT(i,0));
        bVppp(span(iN, iN+TT(i,0).size()-1)) = B.elem(TT(i,0));
        cVppp(span(iN, iN+TT(i,0).size()-1)) = A.elem(TT(i,1));
        dVppp(span(iN, iN+TT(i,0).size()-1)) = B.elem(TT(i,1));
        iN += TT(i,0).size();
    }


    cout << "Good so far... (6)"  << (double)(clock() - t)/CLOCKS_PER_SEC<< endl;
    t = clock();


    //Interaction is currently precalculated here and stored in a sparse matrix
    vec vValsVppp = V3(aVppp+iNh,bVppp+iNh,cVppp+iNh,dVppp+iNh); //this works, tested agains bs.v2, 9.4.2015
    iN = vValsVppp.size();




    cout << "Good so far... (7)"  << (double)(clock() - t)/CLOCKS_PER_SEC<< endl;
    //cout << "Number of nonzeros:" << vValsVppp(find(vValsVppp==0.0)).size() << endl;
    //cout << "Number of nonzeros:" << find(vValsVppp).size() << endl;


    t = clock();
    //use symmetries to fill in remaining interactions

    aVpppp.set_size(4*iN);
    bVpppp.set_size(4*iN);
    cVpppp.set_size(4*iN);
    dVpppp.set_size(4*iN);

    aVpppp(span(0,iN-1)) = aVppp;
    aVpppp(span(iN,2*iN-1)) = bVppp;
    aVpppp(span(2*iN,3*iN-1)) = cVppp;
    aVpppp(span(3*iN,4*iN-1)) = dVppp;

    bVpppp(span(0,iN-1)) = bVppp;
    bVpppp(span(iN,2*iN-1)) = aVppp;
    bVpppp(span(2*iN,3*iN-1)) = dVppp;
    bVpppp(span(3*iN,4*iN-1)) = cVppp;

    cVpppp(span(0,iN-1)) = cVppp;
    cVpppp(span(iN,2*iN-1)) = dVppp;
    cVpppp(span(2*iN,3*iN-1)) = bVppp;
    cVpppp(span(3*iN,4*iN-1)) = aVppp;

    dVpppp(span(0,iN-1)) = dVppp;
    dVpppp(span(iN,2*iN-1)) = cVppp;
    dVpppp(span(2*iN,3*iN-1)) = aVppp;
    dVpppp(span(3*iN,4*iN-1)) = bVppp;

    cout << "Good so far... (8)"  << (double)(clock() - t)/CLOCKS_PER_SEC<< endl;
    t = clock();


    vValsVpppp.set_size(4*iN);
    vValsVpppp(span(0,iN-1)) = vValsVppp;
    vValsVpppp(span(iN,2*iN-1)) = vValsVppp;
    vValsVpppp(span(2*iN,3*iN-1)) = -vValsVppp;
    vValsVpppp(span(3*iN,4*iN-1)) = -vValsVppp;


    //Testing that the full interaction is correct
    /*
    double val = 0;
    double val2 = 0;
    int disccount = 0;
    int an,bn,cn,dn;
    for(int n =0; n< vValsVpppp.size(); n++){
        an = aVpppp(n) + iNh;
        bn = bVpppp(n) + iNh;
        cn = cVpppp(n) + iNh;
        dn = dVpppp(n) + iNh;

        val = bs.v2(an,bn,cn,dn);

        if((abs(val - vValsVpppp(n)))>0.00001){
            cout << val << " " << val2 << " " << vValsVpppp(n) << endl;
            disccount += 1;
        }
    }
    */


    //cout << "Discrepancies in Vpppp:" << disccount << endl;
    //cout << "Size of Vpppp         :" << vValsVpppp.size() << endl;

}

void initializer::sVpppp(){
    //cout << "Hello from initializer!" << endl;
    clock_t t;

    Vpppp.set_size(iNp2, iNp2);
    t = clock();

    vec AB = linspace(0,iNp2-1,iNp2);

    uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    uvec A = conv_to<uvec>::from(AB) - B*iNp;

    //vec KAx = bs.vKx.elem(A+iNh);
    //vec KAy = bs.vKy.elem(A+iNh);
    //vec KAz = bs.vKz.elem(A+iNh);

    //vec KBx = bs.vKx.elem(B+iNh);
    //vec KBy = bs.vKy.elem(B+iNh);
    //vec KBz = bs.vKz.elem(B+iNh);

    ivec KABx = bs.vKx.elem(A+iNh)+bs.vKx.elem(B+iNh);
    ivec KABy = iNmax*(bs.vKy.elem(A+iNh)+bs.vKy.elem(B+iNh));
    ivec KABz = iNmax2*(bs.vKz.elem(A+iNh)+bs.vKz.elem(B+iNh));
    ivec KABms = iNmax*iNmax2*(bs.vMs(A+iNh) + bs.vMs(B + iNh));

    ivec KAB = KABx+KABy+KABz + KABms;
    ivec KAB_unique = unique(KAB);
    //cout << "    Stage 1:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();

    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    T0.set_size(0);
    T1.set_size(0);

    //int iN;
    field<uvec> TT;
    TT.set_size(KAB_unique.size(), 2);
    int iN = 0;
    //cout << "    Stage 2:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();


    for(int i = 0; i < KAB_unique.size(); ++i){
        //this is the most time-consuming process in initialization
        //vec T = AB.elem(find(KAB==KAB_unique(i)));
        vec T = conv_to<vec>::from(find(KAB==KAB_unique(i))); //Is it possible to exploit to make this vector should "shrink" ?
        vec O = ones(T.size());
        uvec t0 = conv_to<uvec>::from(kron(T, O));
        uvec t1 = conv_to<uvec>::from(kron(O, T));
        TT(i, 0) = t0;
        TT(i, 1) = t1;
        iN += t0.size();
    }

    //cout << "    Stage 3:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();
    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    //cout << "    Stage 4:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();
    for(int i = 0; i < KAB_unique.size(); ++i){
        T0(span(iN, iN+TT(i,0).size()-1)) = TT(i,0);
        T1(span(iN, iN+TT(i,1).size()-1)) = TT(i,1);
        iN += TT(i,0).size();
    }


    bVpppp = conv_to<uvec>::from(floor(T0/iNp)); //convert to unsigned integer indexing vector
    aVpppp = conv_to<uvec>::from(T0) - bVpppp*iNp ;
    dVpppp = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    cVpppp = conv_to<uvec>::from(T1) - dVpppp*iNp;
    //cout << "    Stage 6:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();

    vValsVpppp = V3(aVpppp+iNh,bVpppp+iNh,cVpppp+iNh,dVpppp+iNh); //this works, tested agains bs.v2, 9.8.2015


    /*
    double val = 0;
    double val2 = 0;
    int disccount = 0;
    int an,bn,cn,dn;
    for(int n =0; n< vValsVpppp.size(); n++){
        an = aVpppp(n) + iNh;
        bn = bVpppp(n) + iNh;
        cn = cVpppp(n) + iNh;
        dn = dVpppp(n) + iNh;

        val = bs.v2(an,bn,cn,dn);

        if((abs(val - vValsVpppp(n)))>0.00001){
            cout << val << " " << val2 << " " << vValsVpppp(n) << endl;
            disccount += 1;
        }
    }

    cout << "Discrepancies in Vpppp:" << disccount << endl;
    cout << "Size of Vpppp         :" << vValsVpppp.size() << endl;
    */
}

void initializer::sVhhhhO(){
    //optimized interaction for Vpppp

    //using symmetries, only consider a>b>:
    uvec I, J; //vectors containing indices (row and column)
    I.set_size(iNh*((iNh+1.0)/2.0));
    J.set_size(iNh*((iNh+1.0)/2.0));

    uint n = 0;
    for(uint i = 0; i<iNh; ++i){
        for(uint j = i; j<iNh; ++j){
            I(n) = i;
            J(n) = j;
            n += 1;
        }
    }


    //Setting up a vector containint a unique integer identifier for K + M_s
    ivec KIJx = bs.vKx.elem(I)+bs.vKx.elem(J);
    ivec KIJy = iNmax*(bs.vKy.elem(I)+bs.vKy.elem(J));
    ivec KIJz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(J));
    ivec KIJms = iNmax*iNmax2*(bs.vMs(I) + bs.vMs(J));

    ivec KIJ = KIJx+KIJy+KIJz + KIJms;
    ivec KIJ_unique = unique(KIJ);

    field<uvec> TT;
    TT.set_size(KIJ_unique.size(), 2);
    int iN = 0;
    vec T, O;
    uvec t0, t1;


    for(uint i = 0; i < KIJ_unique.size(); ++i){
        //locating non-zero regions where K_a + K_b = K_c + K_d
        //it is possible to exploit spin symmetry further inside this loop
        T = conv_to<vec>::from(find(KIJ==KIJ_unique(i))); //Is it possible to make this vector "shrink" as more indices is identified?
        O = ones(T.size());
        t0 = conv_to<uvec>::from(kron(T, O));
        t1 = conv_to<uvec>::from(kron(O, T));
        TT(i, 0) = t0;
        TT(i, 1) = t1;
        iN += t0.size();
    }

    uvec iVhhh, jVhhh, kVhhh, lVhhh;
    iVhhh.set_size(iN);
    jVhhh.set_size(iN);
    kVhhh.set_size(iN);
    lVhhh.set_size(iN);


    iN = 0;
    for(uint i = 0; i < KIJ_unique.size(); ++i){
        iVhhh(span(iN, iN+TT(i,0).size()-1)) = I.elem(TT(i,0));
        jVhhh(span(iN, iN+TT(i,0).size()-1)) = J.elem(TT(i,0));
        kVhhh(span(iN, iN+TT(i,0).size()-1)) = I.elem(TT(i,1));
        lVhhh(span(iN, iN+TT(i,0).size()-1)) = J.elem(TT(i,1));
        iN += TT(i,0).size();
    }


    vec vValsVhhh = V3(iVhhh,jVhhh,kVhhh,lVhhh); //this works, tested agains bs.v2, 9.4.2015
    iN = vValsVhhh.size();

    //use symmetries to fill in remaining interactions
    iVhhhh.set_size(4*iN);
    jVhhhh.set_size(4*iN);
    kVhhhh.set_size(4*iN);
    lVhhhh.set_size(4*iN);

    iVhhhh(span(0,iN-1)) = iVhhh;
    iVhhhh(span(iN,2*iN-1)) = jVhhh;
    iVhhhh(span(2*iN,3*iN-1)) = kVhhh;
    iVhhhh(span(3*iN,4*iN-1)) = lVhhh;

    jVhhhh(span(0,iN-1)) = jVhhh;
    jVhhhh(span(iN,2*iN-1)) = iVhhh;
    jVhhhh(span(2*iN,3*iN-1)) = lVhhh;
    jVhhhh(span(3*iN,4*iN-1)) = kVhhh;

    kVhhhh(span(0,iN-1)) = kVhhh;
    kVhhhh(span(iN,2*iN-1)) = lVhhh;
    kVhhhh(span(2*iN,3*iN-1)) = jVhhh;
    kVhhhh(span(3*iN,4*iN-1)) = iVhhh;

    lVhhhh(span(0,iN-1)) = lVhhh;
    lVhhhh(span(iN,2*iN-1)) = kVhhh;
    lVhhhh(span(2*iN,3*iN-1)) = iVhhh;
    lVhhhh(span(3*iN,4*iN-1)) = jVhhh;

    vValsVhhhh.set_size(4*iN);
    vValsVhhhh(span(0,iN-1)) = vValsVhhh;
    vValsVhhhh(span(iN,2*iN-1)) = vValsVhhh;
    vValsVhhhh(span(2*iN,3*iN-1)) = -vValsVhhh;
    vValsVhhhh(span(3*iN,4*iN-1)) = -vValsVhhh;


}



void initializer::sVhhhh(){

    Vhhhh.set_size(iNh2, iNh2);

    vec IJ = linspace(0,iNh2-1,iNh2);

    uvec J = conv_to<uvec>::from(floor(IJ/iNh)); //convert to unsigned integer indexing vector
    uvec I = conv_to<uvec>::from(IJ) - J*iNh;

    ivec KIx = bs.vKx.elem(I);
    ivec KIy = bs.vKy.elem(I);
    ivec KIz = bs.vKz.elem(I);

    ivec KJx = bs.vKx.elem(J);
    ivec KJy = bs.vKy.elem(J);
    ivec KJz = bs.vKz.elem(J);

    ivec KIJx =         bs.vKx.elem(I)+bs.vKx.elem(J);
    ivec KIJy = iNmax* (bs.vKy.elem(I)+bs.vKy.elem(J));
    ivec KIJz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(J));

    ivec KIJ = KIJx+KIJy+KIJz;
    ivec KIJ_unique = unique(KIJ);

    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    //T0.set_size(0);
    //T1.set_size(0);

    field<uvec> TT;
    TT.set_size(KIJ_unique.size(), 2);
    int iN = 0;

    for(int i = 0; i < KIJ_unique.size(); ++i){
        vec T = IJ.elem(find(KIJ==KIJ_unique(i)));
        vec O = ones(T.size());
        uvec t0 = conv_to<uvec>::from(kron(T, O));
        uvec t1 = conv_to<uvec>::from(kron(O, T));

        TT(i, 0) = t0;
        TT(i, 1) = t1;
        iN += t0.size();
    }


    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    int j;
    for(int i = 0; i < KIJ_unique.size(); i++){
        //this is the most time-consuming process in initialization
        //T0(span(iN, iN+TT(i,0).size()-1)) = TT(i,0);
        //T1(span(iN, iN+TT(i,1).size()-1)) = TT(i,1);
        //iN += TT(i,0).size();

        for(j = 0; j < TT(i,0).size(); ++j){
            T0(iN) = TT(i,0)(j);
            T1(iN) = TT(i,1)(j);
            iN += 1;
        }
    }

    jVhhhh = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    iVhhhh = conv_to<uvec>::from(T0) - jVhhhh*iNh ;
    lVhhhh = conv_to<uvec>::from(floor(T1/iNh)) ; //convert to unsigned integer indexing vector
    kVhhhh = conv_to<uvec>::from(T1) - lVhhhh*iNh;

    vValsVhhhh = V3(iVhhhh,jVhhhh,kVhhhh,lVhhhh);



    //umat locations;
    //locations.set_size(T0.size(),2);
    //locations.col(0) = T0;
    //locations.col(1) = T1;
    //Vhhhh = sp_mat(locations.t(), vValsVhhhh, iNh2, iNh2);

    //test for consistency

    /*
    //int disccount = 0;
    for(int n= 0; n<vValsVhhhh.size();n++){
        if(vValsVhhhh(n) !=  bs.v2(iVhhhh(n), jVhhhh(n), kVhhhh(n), lVhhhh(n))){
            //disccount += 1;
            cout << iVhhhh(n) << " " << jVhhhh(n)<< " " << kVhhhh(n)<< " " << lVhhhh(n)<< " " << vValsVhhhh(n)<< " " <<bs.v(iVhhhh(n), jVhhhh(n), kVhhhh(n), lVhhhh(n)) << endl;
        }
    }
    //cout << "Number of discrepancies in Vhhhh:" <<  disccount << endl;

    double d1,d2;
    int discs = 0;
    for(int i = 0; i< iNh; i++){
        for(int j = i; j< iNh; j++){
            for(int k = j; k< iNh; k++){
                for(int l = k; l < iNh; l++){
                    d1 = Vhhhh(i + j*iNh, k + l*iNh);
                    d2 = bs.v2(i,j,k,l);
                    if(d1 != d2){
                        //cout << d1 << " " << d2 << endl;
                        discs += 1;

                    }
                }
            }
        }
    }
    */

    //cout << "Discrepancy found in Vhhhh:" << discs << endl;

}

void initializer::sVhhpp(){
    Vhhpp.set_size(iNh2, iNp2);

    //indexing rows

    vec IJ = linspace(0,iNh2-1,iNh2);

    uvec J = conv_to<uvec>::from(floor(IJ/iNh)); //convert to unsigned integer indexing vector
    uvec I = conv_to<uvec>::from(IJ) - J*iNh;

    ivec KIx = bs.vKx.elem(I);
    ivec KIy = bs.vKy.elem(I);
    ivec KIz = bs.vKz.elem(I);

    ivec KJx = bs.vKx.elem(J);
    ivec KJy = bs.vKy.elem(J);
    ivec KJz = bs.vKz.elem(J);

    ivec KIJx = bs.vKx.elem(I)+bs.vKx.elem(J);
    ivec KIJy = iNmax*(bs.vKy.elem(I)+bs.vKy.elem(J));
    ivec KIJz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(J));

    ivec KIJ = KIJx+KIJy+KIJz;
    ivec KIJ_unique = unique(KIJ);

    //indexing columns

    vec AB = linspace(0,iNp2-1,iNp2);

    uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    uvec A = conv_to<uvec>::from(AB) - B*iNp;

    ivec KAx = bs.vKx.elem(A+iNh);
    ivec KAy = bs.vKy.elem(A+iNh);
    ivec KAz = bs.vKz.elem(A+iNh);

    ivec KBx = bs.vKx.elem(B+iNh);
    ivec KBy = bs.vKy.elem(B+iNh);
    ivec KBz = bs.vKz.elem(B+iNh);

    ivec KABx = bs.vKx.elem(A+iNh)+bs.vKx.elem(B+iNh);
    ivec KABy = iNmax*(bs.vKy.elem(A+iNh)+bs.vKy.elem(B+iNh));
    ivec KABz = iNmax2*(bs.vKz.elem(A+iNh)+bs.vKz.elem(B+iNh));

    ivec KAB = KABx+KABy+KABz;
    ivec KAB_unique = unique(KAB);

    //consolidating rows and columns

    ivec K_joined = join_cols<imat>(KIJ_unique, KAB_unique);
    //K_joined.set_size(KAB_unique.size() + KIJ_unique.size());
    //K_joined(span(0,KAB_unique.size()-1)) = KAB_unique;
    //K_joined(span(KAB_unique.size(),KIJ_unique.size())) = KIJ_unique;

    ivec K_unique = unique(K_joined);



    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    T0.set_size(0);
    T1.set_size(0);

    field<uvec> TT;
    TT.set_size(K_unique.size(), 2);
    int iN = 0;

    for(int i = 0; i < K_unique.size(); ++i){
        vec Tij = IJ.elem(find(KIJ==K_unique(i)));
        vec ONh = ones(Tij.size());

        vec Tab = AB.elem(find(KAB==K_unique(i)));
        vec ONp = ones(Tab.size());


        if(Tij.size() != 0 && Tab.size() != 0){
            uvec t0 = conv_to<uvec>::from(kron(Tij, ONp));
            uvec t1 = conv_to<uvec>::from(kron(ONh, Tab));
            //T0 = append(T0,t0);
            //T1 = append(T1,t1);
            TT(i, 0) = t0;
            TT(i, 1) = t1;
            iN += t0.size();

        }
    }

    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    int j;
    for(int i = 0; i < K_unique.size(); ++i){
        //this is the most time-consuming process in initialization of Vhhpp
        if(TT(i,0).size() != 0){
            T0(span(iN, iN+TT(i,0).size()-1)) = TT(i,0);
            T1(span(iN, iN+TT(i,1).size()-1)) = TT(i,1);
            iN += TT(i,0).size();}
        //for(j = 0; j < TT(i,0).size(); j++){
        //    T0(iN) = TT(i,0)(j);
        //    T1(iN) = TT(i,1)(j);
        //    iN += 1;
        //}
    }

    jVhhpp = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    iVhhpp = conv_to<uvec>::from(T0) - jVhhpp*iNh ;
    bVhhpp = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    aVhhpp = conv_to<uvec>::from(T1) - bVhhpp*iNp;
    //cout << "Maximum a:" << aVhhpp.max() << endl;

    iVpphh = iVhhpp;
    jVpphh = jVhhpp;
    aVpphh = aVhhpp;
    bVpphh = bVhhpp;

    vValsVhhpp = V3(iVhhpp,jVhhpp,aVhhpp+iNh,bVhhpp+iNh);
    vValsVpphh = V3(aVhhpp+iNh,bVhhpp+iNh,iVhhpp,jVhhpp); //Symmetric? Do a test

    //umat locations;
    //locations.set_size(T0.size(),2);
    //locations.col(0) = T0;
    //locations.col(1) = T1;
    //Vhhpp = sp_mat(locations.t(), vValsVhhpp, iNh2, iNp2);

    //locations.col(0) = T1;
    //locations.col(1) = T0;
    //fmVpphh = flexmat(a,b,i,j,iNp,iNh);

    //cout << V(conv_to<uvec>::from(ones(2)*13),conv_to<uvec>::from(ones(2)*12),conv_to<uvec>::from(ones(2)*46),conv_to<uvec>::from(ones(2)*42)) << endl; //this element is not inlcuded in the testing above!!!
    //cout << Vhhpp(181, 1152) << endl;

    /*
    double val = 0;
    double val2 = 0;
    int disccount = 0;
    int an,bn,in,jn;

    for(int n =0; n< vValsVhhpp.size(); n++){
        //cout << aVhhpp(n) + iNh << endl;
        an = aVhhpp(n) + iNh;
        bn = bVhhpp(n) + iNh;
        in = iVhhpp(n);
        jn = jVhhpp(n);
        //cout << an << endl;

        val = bs.v2(an,bn,in,jn);

        if((abs(val - vValsVpphh(n)))>0.00001){
            cout << val << " " << vValsVpphh(n) << endl;
            disccount += 1;
        }
    }
    cout << "Discrepancies in Vpphh:" << disccount << endl;


    val = 0;
    for(int i = 0; i<iNh; i++){
        for(int j = 0; j<iNh; j++){
            for(int a = 0; a<iNp; a++){
                for(int b = 0; b<iNp; b++){
                    val = bs.v2(a+iNh,b+iNh,i,j);
                    //cout << Vhhpp(i + j*iNh, a + b*iNp) << "       " << val << endl;
                    //cout << Vhhpp(i + j*iNh, a + b*iNp) << endl;
                    //cout << i + j*iNh << " " << a + b*iNp << endl;
                    if(abs(Vhhpp(i + j*iNh, a + b*iNp) - val)>0.00000001){
                        //cout << i + j*iNh << " " << a + b*iNp << endl;
                        cout << i << " " << j << " " << a  << " " << b  << endl;
                        cout << Vhhpp(i + j*iNh, a + b*iNp) - val << endl;
                    }
                }
            }
        }
    }
    */


}

void initializer::sVpphh(){
}

void initializer::sVhpph(){
    Vhpph.set_size(iNhp, iNhp);

    //indexing rows

    vec IA = linspace(0,iNhp-1,iNhp);

    uvec A = conv_to<uvec>::from(floor(IA/iNh)); //convert to unsigned integer indexing vector
    uvec I = conv_to<uvec>::from(IA) - A*iNh;

    ivec KIx = bs.vKx.elem(I);
    ivec KIy = bs.vKy.elem(I);
    ivec KIz = bs.vKz.elem(I);

    ivec KAx = bs.vKx.elem(A+iNh);
    ivec KAy = bs.vKy.elem(A+iNh);
    ivec KAz = bs.vKz.elem(A+iNh);

    ivec KIAx = bs.vKx.elem(I)+bs.vKx.elem(A+iNh);
    ivec KIAy = iNmax*(bs.vKy.elem(I)+bs.vKy.elem(A+iNh));
    ivec KIAz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(A+iNh));

    ivec KIA = KIAx+KIAy+KIAz;
    ivec KIA_unique = unique(KIA);

    //indexing columns

    vec BJ = linspace(0,iNhp-1,iNhp); //AB

    uvec J = conv_to<uvec>::from(floor(BJ/iNp)); //convert to unsigned integer indexing vector
    uvec B = conv_to<uvec>::from(BJ) - J*iNp;

    ivec KJx = bs.vKx.elem(J);
    ivec KJy = bs.vKy.elem(J);
    ivec KJz = bs.vKz.elem(J);

    ivec KBx = bs.vKx.elem(B+iNh);
    ivec KBy = bs.vKy.elem(B+iNh);
    ivec KBz = bs.vKz.elem(B+iNh);

    ivec KBJx = bs.vKx.elem(B+iNh)+bs.vKx.elem(J);
    ivec KBJy = iNmax*(bs.vKy.elem(B+iNh)+bs.vKy.elem(J));
    ivec KBJz = iNmax2*(bs.vKz.elem(B+iNh)+bs.vKz.elem(J));

    ivec KBJ = KBJx+KBJy+KBJz;
    ivec KBJ_unique = unique(KBJ);

    //consolidating rows and columns

    ivec K_joined = join_cols<imat>(KIA_unique, KBJ_unique);
    //K_joined.set_size(KAB_unique.size() + KIJ_unique.size());
    //K_joined(span(0,KAB_unique.size()-1)) = KAB_unique;
    //K_joined(span(KAB_unique.size(),KIJ_unique.size())) = KIJ_unique;

    ivec K_unique = unique(K_joined);



    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    T0.set_size(0);
    T1.set_size(0);


    field<uvec> TT;
    TT.set_size(K_unique.size(), 2);
    int iN = 0;
    //int iN;
    for(int i = 0; i < K_unique.size(); ++i){
        vec Tia = IA.elem(find(KIA==K_unique(i)));
        vec ONh = ones(Tia.size());

        vec Tbj = BJ.elem(find(KBJ==K_unique(i)));
        vec ONp = ones(Tbj.size());

        //153 1586
        //uvec t0 = conv_to<uvec>::from(kron(Tij, ONp));
        //uvec t1 = conv_to<uvec>::from(kron(ONh, Tab));
        //T0 = append(T0,t0);
        //T1 = append(T1,t1);

        if(Tia.size() != 0 && Tbj.size() != 0){
            uvec t0 = conv_to<uvec>::from(kron(Tia, ONp));
            uvec t1 = conv_to<uvec>::from(kron(ONh, Tbj));
            //T0 = append(T0,t0);
            //T1 = append(T1,t1);
            TT(i, 0) = t0;
            TT(i, 1) = t1;
            iN += t0.size();
        }
    }

    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    for(int i = 0; i < K_unique.size(); ++i){
        //this is the most time-consuming process in initialization
        if(TT(i,0).size() != 0){
            T0(span(iN, iN+TT(i,0).size()-1)) = TT(i,0);
            T1(span(iN, iN+TT(i,1).size()-1)) = TT(i,1);
            iN += TT(i,0).size();}
    }


    aVhpph = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    iVhpph = conv_to<uvec>::from(T0) - aVhpph*iNh ;
    jVhpph = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    bVhpph = conv_to<uvec>::from(T1) - jVhpph*iNp;

    vValsVhpph = V3(iVhpph,aVhpph+iNh,bVhpph+iNh,jVhpph);
    //umat locations;
    //locations.set_size(T0.size(),2);
    //locations.col(0) = T0;
    //locations.col(1) = T1;
    //Vhpph = sp_mat(locations.t(), vValsVhpph, iNhp, iNhp);

    //cout << V(conv_to<uvec>::from(ones(2)*13),conv_to<uvec>::from(ones(2)*12),conv_to<uvec>::from(ones(2)*46),conv_to<uvec>::from(ones(2)*42)) << endl; //this element is not inlcuded in the testing above!!!
    //cout << Vhhpp(181, 1152) << endl;

    /*


    double val = 0;
    for(int i = 0; i<iNh; i++){
        for(int j = 0; j<iNh; j++){
            for(int a = 0; a<iNp; a++){
                for(int b = 0; b<iNp; b++){
                    val = bs.v2(i,a+iNh,b+iNh,j);
                    //cout << Vhhpp(i + j*iNh, a + b*iNp) << "       " << val << endl;
                    //cout << Vhhpp(i + j*iNh, a + b*iNp) << endl;
                    //cout << i + j*iNh << " " << a + b*iNp << endl;
                    if(abs(Vhpph(i + a*iNh, b + j*iNp) - val)>0.000001){
                        //cout << i + j*iNh << " " << a + b*iNp << endl;
                        cout << i << " " << a << " " << b  << " " << j  << endl;
                        cout << Vhpph(i + a*iNh, b + j*iNp) << " " << val << endl;
                    }
                }
            }
        }
    }
    */


}


