#include "initializer.h"
#include "basis/electrongas.h"
#define ARMA_64BIT_WORD
#include <armadillo>

using namespace std;
using namespace arma;

initializer::initializer(electrongas Bs)
{
    bs = Bs;
    iNp = bs.iNbstates-bs.iNparticles; //number of particle states
    iNh = bs.iNparticles; //number of hole states
    iNp2 = iNp*iNp;
    iNmax = 2*(bs.vKx.max()+1);
    iNmax2 = iNmax*iNmax;
}

vec initializer::V(uvec t0, uvec t1){
    //delta function of summation of momentum quantum numbers is assumed to have passed before entering here
    uvec b = conv_to<uvec>::from(floor(t0/iNp)); //convert to unsigned integer indexing vector
    uvec a = conv_to<uvec>::from(t0) - b*iNp;
    uvec d = conv_to<uvec>::from(floor(t1/iNp)); //convert to unsigned integer indexing vector
    uvec c = conv_to<uvec>::from(t1) - d*iNp;

    vec ret = zeros(t0.size());

    vec Msa = bs.vMs.elem(a+iNh);
    vec Msb = bs.vMs.elem(b+iNh);
    vec Msc = bs.vMs.elem(c+iNh);
    vec Msd = bs.vMs.elem(d+iNh);

    vec Kax = bs.vKx.elem(a+iNh);
    vec Kay = bs.vKy.elem(a+iNh);
    vec Kaz = bs.vKz.elem(a+iNh);

    vec Kbx = bs.vKx.elem(b+iNh);
    vec Kby = bs.vKy.elem(b+iNh);
    vec Kbz = bs.vKz.elem(b+iNh);

    vec Kcx = bs.vKx.elem(c+iNh);
    vec Kcy = bs.vKy.elem(c+iNh);
    vec Kcz = bs.vKz.elem(c+iNh);

    vec Kdx = bs.vKx.elem(d+iNh);
    vec Kdy = bs.vKy.elem(d+iNh);
    vec Kdz = bs.vKz.elem(d+iNh);


    vec KDplus = ones(t0.size()); //*4*pi/bs.dL3;
    KDplus = KDplus%(Kax+Kbx==Kcx+Kdx)%(Kay+Kby==Kcy+Kdy)%(Kaz+Kbz==Kcz+Kdz);
    //KDplus.elem(conv_to<uvec>::from(find(Kax+Kbx!=Kcx+Kdx || Kay+Kby!=Kcy+Kdy || Kaz+Kbz!=Kcz+Kdz))) *= 0;

    vec diff_ca = absdiff2(Kcx, Kcy, Kcz, Kax, Kay, Kaz);
    vec diff_da = absdiff2(Kdx, Kdy, Kdz, Kax, Kay, Kaz);

    vec term1 = zeros(t0.size()); //term 1
    term1.elem(conv_to<uvec>::from(find(Msa==Msc && Msb==Msd))) += 4*pi/bs.dL3;
    term1.elem(conv_to<uvec>::from(find(Kax==Kcx && Kay==Kcy && Kaz==Kcz))) *= 0;
    term1.elem(find(term1!=0)) /= 4*pi*pi*diff_ca.elem(find(term1!=0))/bs.dL2;

    vec term2 = zeros(t0.size()); //term 2
    term2.elem(conv_to<uvec>::from(find(Msa==Msd && Msb==Msc))) += 4*pi/bs.dL3;
    term2.elem(conv_to<uvec>::from(find(Kax==Kdx && Kay==Kdy && Kaz==Kdz))) *= 0;
    term2.elem(find(term2!=0)) /= 4*pi*pi*diff_da.elem(find(term2!=0))/bs.dL2;

    return KDplus%(term1 - term2);

}

vec initializer::absdiff2(vec kpx, vec kpy, vec kpz, vec kqx,vec kqy, vec kqz){
    //vectorized absdiff2 |kp - kq|^2
    vec ret1 = kpx-kqx;
    vec ret2 = kpy-kqy;
    vec ret3 = kpz-kqz;
    return ret1%ret1 + ret2%ret2 + ret3%ret3;

}

uvec initializer::append(uvec V1, uvec V2){
    //append vector V2 to V1
    int V1size = V1.size();
    //cout << V1size+V2.size() << endl;
    V1.resize(V1size+V2.size());
    for(int i= 0; i<V2.size(); i++){
        V1(i+V1size) = V2(i);
    }
    return V1;
    //V1(span(V1size,V1.size()-1)) = V2;
}

void initializer::sVpppp(){
    cout << "Hello from initializer!" << endl;
    Vpppp.set_size(iNp2, iNp2);

    vec AB = linspace(0,iNp2-1,iNp2);

    //vec A=AB-floor(AB/iNp)*iNp;
    uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    uvec A = conv_to<uvec>::from(AB) - B*iNp;
    //uvec = B.
    //cout << AB.elem(B) << endl;

    //A.print();
    //B.print();
    vec KAx = bs.vKx.elem(A+iNh);
    vec KAy = bs.vKy.elem(A+iNh);
    vec KAz = bs.vKz.elem(A+iNh);

    vec KBx = bs.vKx.elem(B+iNh);
    vec KBy = bs.vKy.elem(B+iNh);
    vec KBz = bs.vKz.elem(B+iNh);


    vec KABx = bs.vKx.elem(A+iNh)+bs.vKx.elem(B+iNh);
    vec KABy = iNmax*(bs.vKy.elem(A+iNh)+bs.vKy.elem(B+iNh));
    vec KABz = iNmax2*(bs.vKz.elem(A+iNh)+bs.vKz.elem(B+iNh));

    vec KAB = KABx+KABy+KABz;
    vec KAB_unique = unique(KAB);

    uvec T0;// = conv_to<uvec>::from(zeros(0));
    uvec T1;// = conv_to<uvec>::from(zeros(0));
    T0.set_size(0);
    T1.set_size(0);

    //int iN;
    for(int i = 0; i < KAB_unique.size(); i++){
        vec T = AB.elem(find(KAB==KAB_unique(i)));
        vec O = ones(T.size());
        uvec t0 = conv_to<uvec>::from(kron(T, O));
        uvec t1 = conv_to<uvec>::from(kron(O, T));
        T0 = append(T0,t0);
        T1 = append(T1,t1);
    }
    vec values = V(T0, T1);
    umat locations;
    locations.set_size(T0.size(),2);
    locations.col(0) = T0;
    locations.col(1) = T1;
    Vpppp = sp_mat(locations.t(), values);
    //mat X(Vpppp);
    //X.save("vpppp", raw_ascii);
    //Vpppp.save("vpppp_3_rs1_14", raw_ascii);


    double bsv = 0.0;

    for(int i = 0; i<T0.size(); i++){
        int b = floor(T0(i)/iNp);
        int a = T0(i) - b*iNp;
        int d = floor(T1(i)/iNp);
        int c = T1(i) - d*iNp;
        bsv = bs.v2(a+iNh,b+iNh,c+iNh,d+iNh);
        if(Vpppp(a + b*iNp, c + d*iNp) != bsv){
            cout << a << " " << b << " " << c << " " << d << "       " << values(i) << "     " << Vpppp(a + b*iNp, c + d*iNp) << "           " << bsv << endl;
        }
    }



    //KAx.print();

    //uvec KAB = conv_to<uvec>::from(AB).elem(find(KAx==KBx && KAy==KBy &&KAz==KBz));
    //KAB.print();

    //KA.elem(arma::find(KA != 0)) //use only nonzeros

    //AB.print();


    /*
    for(int a = 0; a<iNp;a++){
        for(int b = 0; b<iNp; b++){
            for(int c = 0; c<iNp; c++){
                for(int d = 0; d<iNp; d++){
                    if(Vpppp(a + b*iNp, c + d*iNp) != bs.v(a+iNh,b+iNh,c+iNh,d+iNh)){
                        cout << a << " " << b << " " << c << " " << d << "       " << Vpppp(a + b*iNp, c + d*iNp) << "           " << bs.v(a+iNh,b+iNh,c+iNh,d+iNh) << endl;
                    }
                }
            }
        }
    }
    */

    //Vpppp.print();
    //cout << Vpppp.col_ptrs << endl;


    //vec Z;
    //Z.set_size(Np*Np);
    //Z.zeros(200);
}
