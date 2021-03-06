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
    iNmax =2*bs.vKx.max()+3;
    iNmax2 = iNmax*iNmax;
}



vec initializer::V(uvec p, uvec q, uvec r, uvec s){
    //delta function of summation of momentum quantum numbers is assumed to have passed before entering here
    vec ret; // = zeros(p.n_rows);
    ret.set_size(p.n_rows);

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
    ivec KDplus = conv_to<ivec>::from(ones(p.n_rows)); //*4*pi/bs.dL3;
    KDplus = KDplus%(Kax+Kbx==Kcx+Kdx)%(Kay+Kby==Kcy+Kdy)%(Kaz+Kbz==Kcz+Kdz);

    ivec diff_ca = absdiff2(Kcx, Kcy, Kcz, Kax, Kay, Kaz);
    ivec diff_da = absdiff2(Kdx, Kdy, Kdz, Kax, Kay, Kaz);

    vec term1 = zeros(p.n_rows); //term 1
    //term1.set_size(p.n_rows);
    term1.elem(conv_to<uvec>::from(find(Msa==Msc && Msb==Msd))) += 4*pi/bs.dL3; //By changing this i get comparable results
    term1.elem(conv_to<uvec>::from(find(Kax==Kcx && Kay==Kcy && Kaz==Kcz))) *= 0;
    term1.elem(find(term1!=0)) /= 4*pi*pi*conv_to<vec>::from(diff_ca.elem(find(term1!=0)))/bs.dL2;

    vec term2 = zeros(p.n_rows); //term 2
    //term2.set_size(p.n_rows);
    term2.elem(conv_to<uvec>::from(find(Msa==Msd && Msb==Msc))) += 4*pi/bs.dL3;
    term2.elem(conv_to<uvec>::from(find(Kax==Kdx && Kay==Kdy && Kaz==Kdz))) *= 0;
    term2.elem(find(term2!=0)) /= 4*pi*pi*conv_to<vec>::from(diff_da.elem(find(term2!=0)))/bs.dL2;

    return KDplus%(term1 - term2);

}

vec initializer::V3(uvec p, uvec q, uvec r, uvec s){
    //Inefficient interaction calculation, not really vectorized but returns a vector
    vec vVals;
    vVals.set_size(p.n_rows);
    for(int n = 0; n< p.n_rows; n++){
        vVals(n) = bs.v3(p(n), q(n), r(n), s(n));
    }
    return vVals;
}

void initializer::V3_count_nonzero(uvec p, uvec q, uvec r, uvec s){
    //Count nonzero entries in pqrs
    uint count = 0;
    double val;
    for(int n = 0; n< p.n_rows; n++){
        val = bs.v2(p(n), q(n), r(n), s(n));
        if(val!=0){
            count += 1;
        }
    }
    cout << "Number of nonzero entries in config:" << count << endl;
}

vec initializer::V4(Col<u32> p, Col<u32> q, Col<u32> r, Col<u32> s){
    //Inefficient interaction calculation, not really vectorized but returns a vector
    arma::u32 nnz = p.n_rows;
    double * aux_mem = new double[nnz];
    vec vVals(aux_mem, nnz, false, true);

    for(int n = 0; n< p.n_rows; n++){
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

    vec ret = zeros(t0.n_rows);

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


    ivec KDplus = conv_to<ivec>::from(ones(t0.n_rows)); //*4*pi/bs.dL3;
    KDplus = KDplus%(Kax+Kbx==Kcx+Kdx)%(Kay+Kby==Kcy+Kdy)%(Kaz+Kbz==Kcz+Kdz);

    ivec diff_ca = absdiff2(Kcx, Kcy, Kcz, Kax, Kay, Kaz);
    ivec diff_da = absdiff2(Kdx, Kdy, Kdz, Kax, Kay, Kaz);

    vec term1 = zeros(t0.n_rows); //term 1
    //term1.elem(conv_to<uvec>::from(find(Msa==Msc && Msb==Msd))) += 4*pi/bs.dL3;
    //term1.elem(conv_to<uvec>::from(find(Kax==Kcx && Kay==Kcy && Kaz==Kcz))) *= 0;
    //term1.elem(find(term1!=0)) /= 4*pi*pi*diff_ca.elem(find(term1!=0))/bs.dL2;

    vec term2 = zeros(t0.n_rows); //term 2
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
    int V1size = V1.n_rows;
    V1.resize(V1size+V2.n_rows);
    for(int i= 0; i<V2.n_rows; i++){
        V1(i+V1size) = V2(i);
    }
    return V1;
}


vec initializer::appendvec(vec V1, vec V2){
    //int V1size = V1.n_rows;
    vec V3;
    V3.set_size(V1.n_rows + V2.n_rows);
    for(int i= 0; i<V1.n_rows; i++){
        V3(i) = V1(i);
    }
    for(int i= V1.n_rows; i<V1.n_rows+V2.n_rows; i++){
        V3(i) = V2(i-V1.n_rows);
    }

    return V3;
}


/*
 * The CCDT specific interactions
 *
 */


void initializer::sVhphh(){
    //indexing rows

    vec IA = linspace(0,iNh*iNp-1,iNh*iNp);

    uvec A = conv_to<uvec>::from(floor(IA/iNh));
    uvec I = conv_to<uvec>::from(IA) - A*iNh;

    ivec KIAx = bs.vKx.elem(I)+bs.vKx.elem(A+iNh);
    ivec KIAy = iNmax*(bs.vKy.elem(I)+bs.vKy.elem(A+iNh));
    ivec KIAz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(A+iNh));

    ivec KIA = KIAx+KIAy+KIAz;
    ivec KIA_unique = unique(KIA);

    //indexing columns

    vec JK = linspace(0,iNh*iNh-1,iNh*iNh);

    uvec K = conv_to<uvec>::from(floor(JK/iNh));
    uvec J = conv_to<uvec>::from(JK) - K*iNh;

    ivec KJKx = bs.vKx.elem(J)+bs.vKx.elem(K);
    ivec KJKy = iNmax*(bs.vKy.elem(J)+bs.vKy.elem(K));
    ivec KJKz = iNmax2*(bs.vKz.elem(J)+bs.vKz.elem(K));

    ivec KJK = KJKx+KJKy+KJKz;
    ivec KJK_unique = unique(KJK);

    //consolidating rows and columns
    ivec K_joined = join_cols<imat>(KIA_unique, KJK_unique);
    ivec K_unique = unique(K_joined);

    int iN = 0;
    uvec Tia, Tjk;

    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));

    field<uvec> TT;
    TT.set_size(K_unique.n_rows, 2);

    for(int i = 0; i < K_unique.n_rows; ++i){
        vec Tia = IA.elem(find(KIA==K_unique(i)));
        vec ONh = ones(Tia.n_rows);

        vec Tjk = JK.elem(find(KJK==K_unique(i)));
        vec ONp = ones(Tjk.n_rows);

        if(Tia.n_rows != 0 && Tjk.n_rows != 0){
            uvec t0 = conv_to<uvec>::from(kron(Tia, ONp));
            uvec t1 = conv_to<uvec>::from(kron(ONh, Tjk));
            TT(i, 0) = t0;
            TT(i, 1) = t1;
            iN += t0.n_rows;
        }
    }

    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    for(int i = 0; i < K_unique.n_rows; ++i){
        //this is the most time-consuming process in initialization
        if(TT(i,0).n_rows != 0){
            T0(span(iN, iN+TT(i,0).n_rows-1)) = TT(i,0);
            T1(span(iN, iN+TT(i,1).n_rows-1)) = TT(i,1);
            iN += TT(i,0).n_rows;}
    }


    aVhphh = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    iVhphh = conv_to<uvec>::from(T0) - aVhphh*iNh ;
    kVhphh = conv_to<uvec>::from(floor(T1/iNh)) ; //convert to unsigned integer indexing vector
    jVhphh = conv_to<uvec>::from(T1) - kVhphh*iNh;
    vValsVhphh = V3(iVhphh,aVhphh+iNh,jVhphh,kVhphh);
    //V3_count_nonzero(iVhphh,aVhphh+iNh,jVhphh,kVhphh);
    //cout << "Max " << vValsVhphh.max() << endl;

    //iVhhhp = jVhphh;
    //jVhhhp = kVhphh;
    //kVhhhp = iVhphh;
    //aVhhhp = aVhphh;

    aVhhhp = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    kVhhhp = conv_to<uvec>::from(T0) - aVhhhp*iNh ;
    iVhhhp = conv_to<uvec>::from(floor(T1/iNh)) ; //convert to unsigned integer indexing vector
    jVhhhp = conv_to<uvec>::from(T1) - iVhhhp*iNh;

    vValsVhhhp = V3(iVhhhp,jVhhhp, kVhhhp,aVhhhp+iNh);

}


void initializer::sVppphBlock(){

}

void initializer::sVppph(){
    //indexing rows

    vec BC = linspace(0,iNp*iNp-1,iNp*iNp);

    //uvec I = conv_to<uvec>::from(floor(BCI/(iNp*iNp)));
    uvec C = conv_to<uvec>::from(floor(BC/iNp));
    uvec B = conv_to<uvec>::from(BC) - C*iNp;


    ivec KBCx = bs.vKx.elem(C+iNh)+bs.vKx.elem(B+iNh);
    ivec KBCy = iNmax*(bs.vKy.elem(C+iNh)+bs.vKy.elem(B+iNh));
    ivec KBCz = iNmax2*(bs.vKz.elem(C+iNh)+bs.vKz.elem(B+iNh));
    ivec KBCs = iNmax2*iNmax*(bs.vMs.elem(C+iNh)+bs.vMs.elem(B+iNh));

    ivec KBC = bs.unique(C+iNh) + bs.unique(B+iNh);

    //ivec KBC = KBCx+KBCy+KBCz; // +KBCs;
    ivec KBC_unique = unique(KBC);

    //indexing columns

    vec DK =  linspace(0,iNp*iNh-1,iNp*iNh);



    uvec K = conv_to<uvec>::from(floor(DK/iNp));
    uvec D = conv_to<uvec>::from(DK) - K*iNp;

    ivec KDKx = bs.vKx.elem(D+iNh)+bs.vKx.elem(K);
    ivec KDKy = iNmax*(bs.vKy.elem(D+iNh)+bs.vKy.elem(K));
    ivec KDKz = iNmax2*(bs.vKz.elem(D+iNh)+bs.vKz.elem(K));
    ivec KDKs = iNmax2*iNmax*(bs.vMs.elem(D+iNh)+bs.vMs.elem(K));

    //ivec KDK = KDKx+KDKy+KDKz; //+KDKs;
    ivec KDK = bs.unique(D+iNh) + bs.unique(K);

    ivec KDK_unique = unique(KDK);

    //consolidating rows and columns
    ivec K_joined = join_cols<imat>(KBC_unique, KDK_unique);
    ivec K_unique = unique(K_joined);

    int iN = 0;
    //uvec Tab, Tci;

    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));

    field<uvec> TT;
    TT.set_size(K_unique.n_rows, 2);

    for(int i = 0; i < K_unique.n_rows; ++i){
        vec Tab = BC.elem(find(KBC==K_unique(i)));
        vec ONh = ones(Tab.n_rows);

        vec Tci = DK.elem(find(KDK==K_unique(i)));
        vec ONp = ones(Tci.n_rows);

        if(Tab.n_rows != 0 && Tci.n_rows != 0){
            //Tab.print();
            //cout << endl;
            //Tci.print();
            //cout << endl;
            uvec t0 = conv_to<uvec>::from(kron(Tab, ONp));
            uvec t1 = conv_to<uvec>::from(kron(ONh, Tci));
            TT(i, 0) = t0;
            TT(i, 1) = t1;
            iN += t0.n_rows;
        }
    }

    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    for(int i = 0; i < K_unique.n_rows; ++i){
        //this is the most time-consuming process in initialization
        if(TT(i,0).n_rows != 0){
            T0(span(iN, iN+TT(i,0).n_rows-1)) = TT(i,0);
            T1(span(iN, iN+TT(i,1).n_rows-1)) = TT(i,1);
            iN += TT(i,0).n_rows;}
    }


    cVppph = conv_to<uvec>::from(floor(T0/iNp)) ; //convert to unsigned integer indexing vector
    bVppph = conv_to<uvec>::from(T0) - cVppph*iNp;

    iVppph = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    aVppph = conv_to<uvec>::from(T1) - iVppph*iNp;

    //bVppph.print();





    //iVppph.print();
    vValsVppph = V3(bVppph+iNh,cVppph+iNh,aVppph+iNh,iVppph);


    iVphpp = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    aVphpp = conv_to<uvec>::from(T1) - iVphpp*iNp;

    cVphpp = conv_to<uvec>::from(floor(T0/iNp)) ; //convert to unsigned integer indexing vector
    bVphpp = conv_to<uvec>::from(T0) - cVphpp*iNp;

    vValsVphpp = V3(aVphpp+iNh,iVphpp, bVphpp+iNh,cVphpp+iNh);

    //cout << sum(abs(vValsVppph)) << endl;
}

void initializer::sVhppp(){
    //indexing rows

    vec IA = linspace(0,iNh*iNp-1,iNh*iNp);

    uvec A = conv_to<uvec>::from(floor(IA/iNh));
    uvec I = conv_to<uvec>::from(IA) - A*iNh;

    ivec KIAx = bs.vKx.elem(I)+bs.vKx.elem(A+iNh);
    ivec KIAy = iNmax*(bs.vKy.elem(I)+bs.vKy.elem(A+iNh));
    ivec KIAz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(A+iNh));

    ivec KIA = KIAx+KIAy+KIAz;
    ivec KIA_unique = unique(KIA);

    //indexing columns

    vec BC = linspace(0,iNp*iNp-1,iNp*iNp);

    uvec C = conv_to<uvec>::from(floor(BC/iNp));
    uvec B = conv_to<uvec>::from(BC) - C*iNp;

    ivec KBCx = bs.vKx.elem(B+iNh)+bs.vKx.elem(C+iNh);
    ivec KBCy = iNmax*(bs.vKy.elem(B+iNh)+bs.vKy.elem(C+iNh));
    ivec KBCz = iNmax2*(bs.vKz.elem(B+iNh)+bs.vKz.elem(C+iNh));

    ivec KBC = KBCx+KBCy+KBCz;
    ivec KBC_unique = unique(KBC);

    //consolidating rows and columns
    ivec K_joined = join_cols<imat>(KIA_unique, KBC_unique);
    ivec K_unique = unique(K_joined);

    int iN = 0;
    uvec Tia, Tbc;

    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));

    field<uvec> TT;
    TT.set_size(K_unique.n_rows, 2);

    for(int i = 0; i < K_unique.n_rows; ++i){
        vec Tia = IA.elem(find(KIA==K_unique(i)));
        vec ONh = ones(Tia.n_rows);

        vec Tbc = BC.elem(find(KBC==K_unique(i)));
        vec ONp = ones(Tbc.n_rows);

        if(Tia.n_rows != 0 && Tbc.n_rows != 0){
            uvec t0 = conv_to<uvec>::from(kron(Tia, ONp));
            uvec t1 = conv_to<uvec>::from(kron(ONh, Tbc));
            TT(i, 0) = t0;
            TT(i, 1) = t1;
            iN += t0.n_rows;
        }
    }

    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    for(int i = 0; i < K_unique.n_rows; ++i){
        //this is the most time-consuming process in initialization
        if(TT(i,0).n_rows != 0){
            T0(span(iN, iN+TT(i,0).n_rows-1)) = TT(i,0);
            T1(span(iN, iN+TT(i,1).n_rows-1)) = TT(i,1);
            iN += TT(i,0).n_rows;}
    }

    aVhppp = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    iVhppp = conv_to<uvec>::from(T0) - aVhppp*iNh ;
    cVhppp = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    bVhppp = conv_to<uvec>::from(T1) - cVhppp*iNp;



    vValsVhppp = V3(iVhppp,aVhppp+iNh,bVhppp+iNh,cVhppp+iNh);
    //cout << vValsVhppp.n_rows << endl;
    //cout << vValsVhppp.max() << endl;
    //V3_count_nonzero(iVhppp,aVhppp+iNh,bVhppp+iNh,cVhppp+iNh);

    //vValsVhppp.print();

    //aVpphp = bVhppp;
    //bVpphp = cVhppp;
    //iVpphp = iVhppp;
    //cVpphp = aVhppp;
    cVpphp = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    iVpphp = conv_to<uvec>::from(T0) - cVpphp*iNh ;
    aVpphp = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    bVpphp = conv_to<uvec>::from(T1) - aVpphp*iNp;
    vValsVpphp = V3(aVpphp+iNh,bVpphp+iNh, iVpphp,cVpphp+iNh);
}



void initializer::sVhpppBlock(){
    //indexing rows

    vec IA = linspace(0,iNhp-1,iNhp);

    uvec A = conv_to<uvec>::from(floor(IA/iNh));
    uvec I = conv_to<uvec>::from(IA) - A*iNh;

    ivec KIAx = bs.vKx.elem(I)+bs.vKx.elem(A+iNh);
    ivec KIAy = iNmax*(bs.vKy.elem(I)+bs.vKy.elem(A+iNh));
    ivec KIAz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(A+iNh));

    ivec KIA = KIAx+KIAy+KIAz;
    ivec KIA_unique = unique(KIA);

    //indexing columns

    vec BC = linspace(0,iNp*iNp-1,iNp*iNp);

    uvec C = conv_to<uvec>::from(floor(BC/iNp));
    uvec B = conv_to<uvec>::from(BC) - C*iNp;

    ivec KBCx = bs.vKx.elem(B+iNh)+bs.vKx.elem(C);
    ivec KBCy = iNmax*(bs.vKy.elem(B+iNh)+bs.vKy.elem(C));
    ivec KBCz = iNmax2*(bs.vKz.elem(B+iNh)+bs.vKz.elem(C));

    ivec KBC = KBCx+KBCy+KBCz;
    ivec KBC_unique = unique(KBC);

    //consolidating rows and columns
    ivec K_joined = join_cols<imat>(KIA_unique, KBC_unique);
    ivec K_unique = unique(K_joined);

    int iN = 0;
    uvec Tia, Tbc;
    //bmVhppp.set_size(K_unique.n_rows, iNp, iNp, iNp, iNp);
    for(int i = 0; i < K_unique.n_rows; ++i){

        Tia = find(KIA==K_unique(i));
        Tbc = find(KBC==K_unique(i));

        //bmVhppp.set_block(i, I.elem(Tia), A.elem(Tia),B.elem(Tbc), C.elem(Tbc));
        iN += Tia.n_rows;
    }
}

void initializer::sVphhp(){
    //this one is basically the same as hpph, with the index transformed
    enable_svphhp = true;
}


/*
 * The CCD specific interactions
 *
 */



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
    TT.set_size(KAB_unique.n_rows, 2);
    u32 iN = 0;
    vec T, O;
    uvec tT;
    uvec t0, t1;

    cout << KAB_unique.n_rows << endl;

    //bmVpppp.set_size(KAB_unique.n_rows, iNp, iNp, iNp, iNp);

    cout << "Good so far... (3)"  << (double)(clock() - t)/CLOCKS_PER_SEC<< endl;
    t = clock();



    for(uint i = 0; i < KAB_unique.n_rows; ++i){
        //locating non-zero regions where K_a + K_b = K_c + K_d
        //it is possible to exploit spin symmetry further inside this loop
        T = conv_to<vec>::from(find(KAB==KAB_unique(i))); //Is it possible to make this vector "shrink" as more indices is identified?

        tT = find(KAB==KAB_unique(i));
        //bmVpppp.set_block(i, A.elem(tT), B.elem(tT),A.elem(tT), B.elem(tT));

        O = ones(T.n_rows);
        t0 = conv_to<uvec>::from(kron(T, O));
        t1 = conv_to<uvec>::from(kron(O, T));
        TT(i, 0) = t0;
        TT(i, 1) = t1;
        iN += t0.n_rows;
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
    for(uint i = 0; i < KAB_unique.n_rows; ++i){
        aVppp(span(iN, iN+TT(i,0).n_rows-1)) = A.elem(TT(i,0));
        bVppp(span(iN, iN+TT(i,0).n_rows-1)) = B.elem(TT(i,0));
        cVppp(span(iN, iN+TT(i,0).n_rows-1)) = A.elem(TT(i,1));
        dVppp(span(iN, iN+TT(i,0).n_rows-1)) = B.elem(TT(i,1));
        iN += TT(i,0).n_rows;
    }


    cout << "Good so far... (6)"  << (double)(clock() - t)/CLOCKS_PER_SEC<< endl;
    t = clock();


    //Interaction is currently precalculated here and stored in a sparse matrix
    vec vValsVppp = V3(aVppp+iNh,bVppp+iNh,cVppp+iNh,dVppp+iNh); //this works, tested agains bs.v2, 9.4.2015
    iN = vValsVppp.n_rows;




    cout << "Good so far... (7)"  << (double)(clock() - t)/CLOCKS_PER_SEC<< endl;
    //cout << "Number of nonzeros:" << vValsVppp(find(vValsVppp==0.0)).n_rows << endl;
    //cout << "Number of nonzeros:" << find(vValsVppp).n_rows << endl;


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

}

void initializer::sVppppBlock_mp(){
    //Block interaction (store only blocks)
    vec AB = linspace(0,iNp2-1,iNp2);
    uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    uvec A = conv_to<uvec>::from(AB) - B*iNp;


    //Setting up a vector containint a unique integer identifier for K + M_s
    ivec KABx = bs.vKx.elem(A+iNh)+bs.vKx.elem(B+iNh);
    ivec KABy = iNmax*(bs.vKy.elem(A+iNh)+bs.vKy.elem(B+iNh));
    ivec KABz = iNmax2*(bs.vKz.elem(A+iNh)+bs.vKz.elem(B+iNh));
    ivec KABms = iNmax*iNmax2*(bs.vMs(A+iNh) + bs.vMs(B + iNh));

    ivec KAB = KABx+KABy+KABz + KABms;
    ivec KAB_unique = unique(KAB);

    field<uvec> TT;
    TT.set_size(KAB_unique.n_rows, 2);
    u32 iN = 0;
    vec T, O;
    //uvec tT;
    uvec t0, t1;


    bmVpppp.set_size(KAB_unique.n_rows, iNp, iNp, iNp, iNp);

    #pragma omp parallel for
    for(uint i = 0; i < KAB_unique.n_rows; ++i){
        //locating non-zero regions where K_a + K_b = K_c + K_d
        //T = conv_to<vec>::from(find(KAB==KAB_unique(i))); //Is it possible to make this vector "shrink" as more indices is identified?
        uvec tT = find(KAB==KAB_unique(i));
        bmVpppp.set_block(i, A.elem(tT), B.elem(tT),A.elem(tT), B.elem(tT));
    }
}


void initializer::sVppppBlock(){
    //Block interaction (store only blocks)
    vec AB = linspace(0,iNp2-1,iNp2);
    uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    uvec A = conv_to<uvec>::from(AB) - B*iNp;


    //Setting up a vector containint a unique integer identifier for K + M_s
    ivec KABx = bs.vKx.elem(A+iNh)+bs.vKx.elem(B+iNh);
    ivec KABy = iNmax*(bs.vKy.elem(A+iNh)+bs.vKy.elem(B+iNh));
    ivec KABz = iNmax2*(bs.vKz.elem(A+iNh)+bs.vKz.elem(B+iNh));
    ivec KABms = iNmax*iNmax2*(bs.vMs(A+iNh) + bs.vMs(B + iNh));

    ivec KAB = KABx+KABy+KABz + KABms;
    ivec KAB_unique = unique(KAB);

    field<uvec> TT;
    TT.set_size(KAB_unique.n_rows, 2);
    u32 iN = 0;
    vec T, O;
    uvec tT;
    uvec t0, t1;


    bmVpppp.set_size(KAB_unique.n_rows, iNp, iNp, iNp, iNp);


    for(uint i = 0; i < KAB_unique.n_rows; ++i){
        //locating non-zero regions where K_a + K_b = K_c + K_d
        //T = conv_to<vec>::from(find(KAB==KAB_unique(i))); //Is it possible to make this vector "shrink" as more indices is identified?
        tT = find(KAB==KAB_unique(i));
        bmVpppp.set_block(i, A.elem(tT), B.elem(tT),A.elem(tT), B.elem(tT));
    }
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
    TT.set_size(KAB_unique.n_rows, 2);
    int iN = 0;
    //cout << "    Stage 2:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();


    for(int i = 0; i < KAB_unique.n_rows; ++i){
        //this is the most time-consuming process in initialization
        //vec T = AB.elem(find(KAB==KAB_unique(i)));
        vec T = conv_to<vec>::from(find(KAB==KAB_unique(i))); //Is it possible to exploit to make this vector should "shrink" ?
        vec O = ones(T.n_rows);
        uvec t0 = conv_to<uvec>::from(kron(T, O));
        uvec t1 = conv_to<uvec>::from(kron(O, T));
        TT(i, 0) = t0;
        TT(i, 1) = t1;
        iN += t0.n_rows;
    }

    //cout << "    Stage 3:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();
    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    //cout << "    Stage 4:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();
    for(int i = 0; i < KAB_unique.n_rows; ++i){
        T0(span(iN, iN+TT(i,0).n_rows-1)) = TT(i,0);
        T1(span(iN, iN+TT(i,1).n_rows-1)) = TT(i,1);
        iN += TT(i,0).n_rows;
    }


    bVpppp = conv_to<uvec>::from(floor(T0/iNp)); //convert to unsigned integer indexing vector
    aVpppp = conv_to<uvec>::from(T0) - bVpppp*iNp ;
    dVpppp = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    cVpppp = conv_to<uvec>::from(T1) - dVpppp*iNp;
    //cout << "    Stage 6:" << (double)(clock() - t)/CLOCKS_PER_SEC << endl;
    //t = clock();

    vValsVpppp = V3(aVpppp+iNh,bVpppp+iNh,cVpppp+iNh,dVpppp+iNh); //this works, tested agains bs.v2, 9.8.2015

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
    TT.set_size(KIJ_unique.n_rows, 2);
    int iN = 0;
    vec T, O;
    uvec t0, t1;


    for(uint i = 0; i < KIJ_unique.n_rows; ++i){
        //locating non-zero regions where K_a + K_b = K_c + K_d
        //it is possible to exploit spin symmetry further inside this loop
        T = conv_to<vec>::from(find(KIJ==KIJ_unique(i))); //Is it possible to make this vector "shrink" as more indices is identified?
        O = ones(T.n_rows);
        t0 = conv_to<uvec>::from(kron(T, O));
        t1 = conv_to<uvec>::from(kron(O, T));
        TT(i, 0) = t0;
        TT(i, 1) = t1;
        iN += t0.n_rows;
    }

    uvec iVhhh, jVhhh, kVhhh, lVhhh;
    iVhhh.set_size(iN);
    jVhhh.set_size(iN);
    kVhhh.set_size(iN);
    lVhhh.set_size(iN);


    iN = 0;
    for(uint i = 0; i < KIJ_unique.n_rows; ++i){
        iVhhh(span(iN, iN+TT(i,0).n_rows-1)) = I.elem(TT(i,0));
        jVhhh(span(iN, iN+TT(i,0).n_rows-1)) = J.elem(TT(i,0));
        kVhhh(span(iN, iN+TT(i,0).n_rows-1)) = I.elem(TT(i,1));
        lVhhh(span(iN, iN+TT(i,0).n_rows-1)) = J.elem(TT(i,1));
        iN += TT(i,0).n_rows;
    }


    vec vValsVhhh = V3(iVhhh,jVhhh,kVhhh,lVhhh); //this works, tested agains bs.v2, 9.4.2015
    iN = vValsVhhh.n_rows;

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
    TT.set_size(KIJ_unique.n_rows, 2);
    int iN = 0;

    for(int i = 0; i < KIJ_unique.n_rows; ++i){
        vec T = IJ.elem(find(KIJ==KIJ_unique(i)));
        vec O = ones(T.n_rows);
        uvec t0 = conv_to<uvec>::from(kron(T, O));
        uvec t1 = conv_to<uvec>::from(kron(O, T));

        TT(i, 0) = t0;
        TT(i, 1) = t1;
        iN += t0.n_rows;
    }


    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    int j;
    for(int i = 0; i < KIJ_unique.n_rows; i++){
        //this is the most time-consuming process in initialization
        //T0(span(iN, iN+TT(i,0).n_rows-1)) = TT(i,0);
        //T1(span(iN, iN+TT(i,1).n_rows-1)) = TT(i,1);
        //iN += TT(i,0).n_rows;

        for(j = 0; j < TT(i,0).n_rows; ++j){
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


}

void initializer::sVhhpp(){
    Vhhpp.set_size(iNh2, iNp2);

    //indexing rows

    vec IJ = linspace(0,iNh2-1,iNh2);

    uvec J = conv_to<uvec>::from(floor(IJ/iNh)); //convert to unsigned integer indexing vector
    uvec I = conv_to<uvec>::from(IJ) - J*iNh;



    ivec KIJx = bs.vKx.elem(I)+bs.vKx.elem(J);
    ivec KIJy = iNmax*(bs.vKy.elem(I)+bs.vKy.elem(J));
    ivec KIJz = iNmax2*(bs.vKz.elem(I)+bs.vKz.elem(J));

    ivec KIJ = KIJx+KIJy+KIJz;
    ivec KIJ_unique = unique(KIJ);

    //indexing columns

    vec AB = linspace(0,iNp2-1,iNp2);

    uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    uvec A = conv_to<uvec>::from(AB) - B*iNp;

    ivec KABx = bs.vKx.elem(A+iNh)+bs.vKx.elem(B+iNh);
    ivec KABy = iNmax*(bs.vKy.elem(A+iNh)+bs.vKy.elem(B+iNh));
    ivec KABz = iNmax2*(bs.vKz.elem(A+iNh)+bs.vKz.elem(B+iNh));

    ivec KAB = KABx+KABy+KABz;
    ivec KAB_unique = unique(KAB);

    //consolidating rows and columns

    ivec K_joined = join_cols<imat>(KIJ_unique, KAB_unique);

    ivec K_unique = unique(K_joined);



    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    T0.set_size(0);
    T1.set_size(0);

    field<uvec> TT;
    TT.set_size(K_unique.n_rows, 2);

    int iN = 0;

    for(int i = 0; i < K_unique.n_rows; ++i){
        vec Tij = IJ.elem(find(KIJ==K_unique(i)));
        vec ONh = ones(Tij.n_rows);

        vec Tab = AB.elem(find(KAB==K_unique(i)));
        vec ONp = ones(Tab.n_rows);

        if(Tij.n_rows != 0 && Tab.n_rows != 0){
            uvec t0 = conv_to<uvec>::from(kron(Tij, ONp));
            uvec t1 = conv_to<uvec>::from(kron(ONh, Tab));

            TT(i, 0) = t0;
            TT(i, 1) = t1;
            iN += t0.n_rows;

        }
    }

    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    int j;
    for(uint i = 0; i < K_unique.n_rows; ++i){
        //this is the most time-consuming process in initialization of Vhhpp
        if(TT(i,0).n_rows != 0){
            T0(span(iN, iN+TT(i,0).n_rows-1)) = TT(i,0);
            T1(span(iN, iN+TT(i,1).n_rows-1)) = TT(i,1);
            iN += TT(i,0).n_rows;}
        //for(j = 0; j < TT(i,0).n_rows; j++){
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

    ivec K_unique = unique(K_joined);



    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    T0.set_size(0);
    T1.set_size(0);


    field<uvec> TT;
    TT.set_size(K_unique.n_rows, 2);
    int iN = 0;
    //int iN;
    for(int i = 0; i < K_unique.n_rows; ++i){
        vec Tia = IA.elem(find(KIA==K_unique(i)));
        vec ONh = ones(Tia.n_rows);

        vec Tbj = BJ.elem(find(KBJ==K_unique(i)));
        vec ONp = ones(Tbj.n_rows);

        if(Tia.n_rows != 0 && Tbj.n_rows != 0){
            uvec t0 = conv_to<uvec>::from(kron(Tia, ONp));
            uvec t1 = conv_to<uvec>::from(kron(ONh, Tbj));
            //T0 = append(T0,t0);
            //T1 = append(T1,t1);
            TT(i, 0) = t0;
            TT(i, 1) = t1;
            iN += t0.n_rows;
        }
    }

    T0.set_size(iN);
    T1.set_size(iN);
    iN = 0;
    for(int i = 0; i < K_unique.n_rows; ++i){
        //this is the most time-consuming process in initialization
        if(TT(i,0).n_rows != 0){
            T0(span(iN, iN+TT(i,0).n_rows-1)) = TT(i,0);
            T1(span(iN, iN+TT(i,1).n_rows-1)) = TT(i,1);
            iN += TT(i,0).n_rows;}
    }


    aVhpph = conv_to<uvec>::from(floor(T0/iNh)); //convert to unsigned integer indexing vector
    iVhpph = conv_to<uvec>::from(T0) - aVhpph*iNh ;
    jVhpph = conv_to<uvec>::from(floor(T1/iNp)) ; //convert to unsigned integer indexing vector
    bVhpph = conv_to<uvec>::from(T1) - jVhpph*iNp;

    vValsVhpph = V3(iVhpph,aVhpph+iNh,bVhpph+iNh,jVhpph);

    if(enable_svphhp){
        //generate vphhp

    }



}


