#include "blockmap.h"
#define ARMA_64BIT_WORD
#include <armadillo>
#include "basis/electrongas.h"

using namespace std;
using namespace arma;

blockmap::blockmap(){}

void blockmap::init(electrongas bs, int n_configs, uvec size){
    eBs = bs;
    k_step = 2*eBs.vKx.max()+3; //stepsize for identifying unique regions
    iNconfigs = n_configs;
    fmBlocks.set_size(iNconfigs); //block of indices

    fmBlockz.set_size(iNconfigs);

    fuvRows.set_size(iNconfigs);
    fuvCols.set_size(iNconfigs);



    fvConfigs.set_size(iNconfigs); //configuration in quantum numbers of each block
    blocklengths.set_size(iNconfigs);  //number of blocks in each configuration
    fmOrdering.set_size(iNconfigs,2); //the ordering of each configuration
    uiCurrent_block = 0;

    Np = eBs.iNbstates-eBs.iNparticles; //conflicting naming here
    Nh = eBs.iNparticles;

    uvSize = size; //true state configurations (Np, Np, Nh, Nh) or (Np, Np, Np, Nh, Nh, Nh) (or similar)

    /*

    uvSize.set_size(4); //particle-hole organization
    uvSize(0) = Np; //rows
    uvSize(1) = Np;
    uvSize(2) = Nh; //columns
    uvSize(3) = Nh;
    */
}

blockmap::blockmap(electrongas bs, int n_configs, uvec size)
{
    eBs = bs;
    k_step = 2*eBs.vKx.max()+3; //stepsize for identifying unique regions
    iNconfigs = n_configs;
    fmBlocks.set_size(iNconfigs); //block of indices

    fmBlockz.set_size(iNconfigs);

    fuvRows.set_size(iNconfigs);
    fuvCols.set_size(iNconfigs);

    fvConfigs.set_size(iNconfigs); //configuration in quantum numbers of each block
    blocklengths.set_size(iNconfigs);  //number of blocks in each configuration
    fmOrdering.set_size(iNconfigs,2); //the ordering of each configuration
    uiCurrent_block = 0;

    Np = eBs.iNbstates-eBs.iNparticles; //conflicting naming here
    Nh = eBs.iNparticles;

    uvSize = size; //true state configurations (Np, Np, Nh, Nh) or (Np, Np, Np, Nh, Nh, Nh) (or similar)

    /*

    uvSize.set_size(4); //particle-hole organization
    uvSize(0) = Np; //rows
    uvSize(1) = Np;
    uvSize(2) = Nh; //columns
    uvSize(3) = Nh;
    */
}



void blockmap::init_interaction(ivec shift){
    //imat L(2,3);
    //imat R(2,3);

    ivec l0 = {0,1,shift(0)};
    ivec l1 = {1,1,shift(1)};
    ivec r0 = {2,1,shift(2)};
    ivec r1 = {3,1,shift(3)};

    imat L = join_rows(l0,l1);
    imat R = join_rows(r0,r1);
    //L.print();
    map_regions(L.t(), R.t());
}

void blockmap::print_block_maximum(){
    //debugging function
    for(uint i = 0; i < blocklengths(0); ++i){
        getblock(0, i).print();
        cout << endl;
    }
}

// ##################################################
// ##                                              ##
// ## Index related functions                      ##
// ##                                              ##
// ##################################################

ivec blockmap::intersect1d(ivec A, ivec B){
    // ###################################################
    // ## Returns the intersection of two unique arrays ##
    // ###################################################
    ivec ret(A.n_rows);
    uint counter = 0;
    for(uint i = 0; i < A.n_rows; ++i){
        int a = A(i);

        for(uint j = 0; j < B.n_rows; ++j){
            if(B(j) == a){
                ret(counter) = a;
                counter += 1;
                break;
            }
        }

    }
    ivec ret2(counter);
    for(uint i = 0; i < counter; ++i){
        ret2(i) = ret(i);
    }
    return ret2;
}

field<uvec> blockmap::unpack(uvec vStream, imat imOrder){
    // ###############################################
    // ## unpack a disorganized sequence of indices ##
    // ###############################################
    int iMsize = imOrder.n_rows+1;
    ivec M(iMsize);
    M(iMsize-1) = 1;
    int mn = 1;
    for(int i = 0; i < iMsize-1; ++i){
        mn *= uvSize(imOrder(i,0));
        M(iMsize - i-2) = mn;
    }
    field<uvec> indices(imOrder.n_rows);
    for(uint i = 0; i < imOrder.n_rows; ++i){
        uvec P = vStream;
        for(uint e = 0; e<i; ++e){
            P -= indices(imOrder.n_rows-e-1)*M(e+1);
        }

        uvec elem = floor(P/M(i+1));
        indices(imOrder.n_rows-i-1) =  elem;
    }
    return indices;
}

uvec blockmap::unpack_uvec(uint vStream, imat imOrder){
    // ###############################################
    // ## unpack a disorganized sequence of indices ##
    // ###############################################
    int iMsize = imOrder.n_rows+1;
    ivec M(iMsize);
    M(iMsize-1) = 1;
    int mn = 1;
    for(int i = 0; i < iMsize-1; ++i){
        mn *= uvSize(imOrder(i,0));
        M(iMsize - i-2) = mn;
    }
    uvec indices(imOrder.n_rows);
    for(uint i = 0; i < imOrder.n_rows; ++i){
        uint P = vStream;
        for(uint e = 0; e<i; ++e){
            P -= indices(imOrder.n_rows - e - 1)*M(e+1);
        }
        indices(imOrder.n_rows - i - 1) = floor(P/M(i+1));
    }
    return indices;
}



uint blockmap::to(uint p, uint q, uint r, uint s){
    return p + q*uvSize(0) + r*uvSize(0)*uvSize(1) + s * uvSize(0)*uvSize(1)*uvSize(2);
}  //compressed index

uvec blockmap::from(uint i){
    uvec ret(4);
    ret(3) = floor(i/(uvSize(0)*uvSize(1)*uvSize(2)));
    ret(2) = floor((i-ret(3)*uvSize(0)*uvSize(1)*uvSize(2))/(uvSize(0)*uvSize(1)));
    ret(1) = floor((i-ret(3)*uvSize(0)*uvSize(1)*uvSize(2) - ret(2)*uvSize(0)*uvSize(1))/uvSize(0));
    ret(0) = i-ret(3)*uvSize(0)*uvSize(1)*uvSize(2) - ret(2)*uvSize(0)*uvSize(1) - ret(1)*uvSize(0);
    return ret;
} //expanded index

// ##################################################
// ##                                              ##
// ## External functions                           ##
// ##                                              ##
// ##################################################

void blockmap::map(ivec left, ivec right){
    imat L(left.n_rows,3);
    //left.print();
    for(uint i = 0; i<left.n_rows; ++i){
        L(i,0) =abs(left(i)) - 1;
        if(left(i)<0){
            L(i,1) = -1;
        }
        else{
            L(i,1) = 1;
        }



        if(uvSize(abs(left(i))-1)==Np){
            L(i,2) = Nh;
        }
        else{
            L(i,2) = 0;
        }
    }

    imat R(right.n_rows,3);
    for(uint i = 0; i<right.n_rows; ++i){
        R(i,0) =abs(right(i)) - 1;
        if(right(i)<0){
            R(i,1) = -1;
        }
        else{
            R(i,1) = 1;
        }
        if(uvSize(abs(right(i))-1)==Np){
            R(i,2) = Nh;
        }
        else{
            R(i,2) = 0;
        }
    }
    map_regions(L,R);

}

field<uvec> blockmap::blocksort_symmetric(ivec K_unique){
    //find blocks in array of the dimensionality Np*Np

    //begin by assembling LHS
    uint Ndim = Np*(Np+1)/2;
    ivec LHS(Ndim);
    uvec A(Ndim);
    uvec B(Ndim);
    uint ncount = 0;
    for(uint a = 0; a< Np; ++a){
        for(uint b= 0; b<=a; ++b){
            //LHS(ncount) = eBs.unique(a) + eBs.unique(b);
            A(ncount) = a;
            B(ncount) = b;
            ncount += 1;
        }
    }
    LHS = eBs.unique(A + Nh) + eBs.unique(B+ Nh);
    if(K_unique.n_rows == 0){
        K_unique = intersect1d(K_unique, unique(LHS));
    }


    uvec l_sorted = sort_index(LHS);
    //uvec l_sorted = linspace<uvec>(0,Ndim-1, Ndim);
    bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool br = false;
    bool row_collect = false;
    int ll_c = l_c;
    uint a;
    uint b;
    uint l_sorted_lc;
    uvec row_a(1000000);
    //uvec row_b(1000000);

    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(l_sorted_lc);

        if(l_c == C){
            a = A(l_sorted_lc);
            b = B(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block

            row_a(nx) = a + b*Np;
            //row_b(nx) = b;
            if(a!=b){
                nx += 1;
                row_a(nx) = b + a*Np;
                //row_b(nx) = a;
            }
            //row(nx) = l_sorted(lc);
            nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                //uvec r = sort(row(span(0,nx-1)));
                tempRows(i) = sort(row_a(span(0,nx-1)));
                //tempRows(1)(i) = row_b(span(0,nx-1));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }

        //cout << row_a(0) << endl;

        lc += 1;
        //cout << l_c - C << endl;
    }

    //collect final block
    tempRows(i) = sort(row_a(span(0,nx-1)));
    //tempRows(1)(i) = row_b(span(0,nx-1));


    return tempRows;

}

field<uvec> blockmap::blocksort(ivec LHS, ivec K_unique){
    uvec l_sorted = sort_index(LHS);
    bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);
    uvec row(1000000);
    uint nx = 0;
    int l_c= LHS(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool row_collect = false;
    int ll_c = l_c;

    while(lc < uiS){
        l_c = LHS(l_sorted(lc));

        if(l_c == C){
            row(nx) = l_sorted(lc);
            nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }


        lc += 1;
        //cout << l_c - C << endl;
    }


    return tempRows;

}

void blockmap::map_vpppp(){
    // ###########################################################
    // ## Special care given to the ladder term                 ##
    // ###########################################################

    /*
    //begin by assembling LHS
    uint Ndim = Np*(Np+1)/2;
    ivec LHS(Ndim);
    uvec A(Ndim);
    uvec B(Ndim);
    uint ncount = 0;
    for(uint a = 0; a< Np; ++a){
        for(uint b= 0; b<=a; ++b){
            //LHS(ncount) = eBs.unique(a) + eBs.unique(b);
            A(ncount) = a;
            B(ncount) = b;
            ncount += 1;
        }
    }
    LHS = eBs.unique(A + Nh) + eBs.unique(B+ Nh);
    ivec K_unique = unique(LHS);


    uvec l_sorted = sort_index(LHS);
    //uvec l_sorted = linspace<uvec>(0,Ndim-1, Ndim);
    bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool br = false;
    bool row_collect = false;
    int ll_c = l_c;
    uint a;
    uint b;
    uint l_sorted_lc;
    uvec row_a(1000000);
    //uvec row_b(1000000);

    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(l_sorted_lc);

        if(l_c == C){
            a = A(l_sorted_lc);
            b = B(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block

            row_a(nx) = a + b*Np;
            //row_b(nx) = b;
            if(a!=b){
                nx += 1;
                row_a(nx) = b + a*Np;
                //row_b(nx) = a;
            }
            //row(nx) = l_sorted(lc);
            nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                //uvec r = sort(row(span(0,nx-1)));
                tempRows(i) = sort(row_a(span(0,nx-1)));
                //tempRows(1)(i) = row_b(span(0,nx-1));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }

        //cout << row_a(0) << endl;

        lc += 1;
        //cout << l_c - C << endl;
    }

    //collect final block
    tempRows(i) = sort(row_a(span(0,nx-1)));
    */




    //fmOrdering(uiCurrent_block,0) = L;
    //fmOrdering(uiCurrent_block,1) = R;

    /*
    uint iNp = uvSize(0);
    uint iNp2 = uvSize(0)*uvSize(0);
    vec AB = linspace(0,iNp2-1,iNp2);

    uvec B = conv_to<uvec>::from(floor(AB/iNp)); //convert to unsigned integer indexing vector
    uvec A = conv_to<uvec>::from(AB) - B*iNp;

    ivec KAB = eBs.unique(A+Nh) + eBs.unique(B+Nh);


    ivec K_unique = unique(KAB);
    */
    //field<uvec> tempRows = blocksort_symmetric({});
    //K_unique = tempRows(tempRows.n_rows);
    //uiN = len(K_unique);
    //uint uiN = K_unique.n_elem;



    field<ivec> lhs = pp();
    ivec K_unique = unique(lhs(2));
    uint uiN = K_unique.n_rows;
    field<uvec> tempRows = partition_pp(lhs, unique(lhs(2)));

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    //fmBlocks(uiCurrent_block).set_size(uiN);
    fmBlockz(uiCurrent_block).set_size(uiN,2);
    fuvCols(uiCurrent_block).set_size(uiN);
    fuvRows(uiCurrent_block).set_size(uiN);

    //field<uvec> tempElements(uiN);
    //field<uvec> tempBlockmap1(uiN);
    //field<uvec> tempBlockmap2(uiN);
    //field<uvec> tempBlockmap3(uiN);

    //uint tempElementsSize = 0;
    //field<uvec> tempRows = blocksort_symmetric({});

    //field
    //field<ivec> lhs = pp();
    //tempRows = partition_pp(lhs, unique(lhs(2)));


    for(uint i = 0; i<uiN; ++i){
        /*
        uvec ind = find(KAB==K_unique(i));
        uvec a = A.elem(ind);
        uvec b = B.elem(ind);
        cout << "a:"<<endl;
        a.print();
        cout << endl;
        //b.print();
        cout << endl;
        */
        uvec i2 = tempRows(i);
        uvec uvB = floor(i2/Np); //convert to unsigned integer indexing vector
        uvec uvA = i2 - uvB*Np;
        //uvA.print();

        //ai.print();
        //cout << endl;


        fmBlockz(uiCurrent_block)(i,0) = uvA; //reusing the framework here, naming does not matter
        fmBlockz(uiCurrent_block)(i,1) = uvB;

        //fmBlockz(uiCurrent_block)(i,0) = tempRows(0)(i); //reusing the framework here, naming does not matter
        //fmBlockz(uiCurrent_block)(i,1) = tempRows(1)(i);


        //fuvRows(uiCurrent_block)(i) = row;
        //fuvCols(uiCurrent_block)(i) = row;


    }

    //lock and load
    uiCurrent_block += 1;




}


void blockmap::map_vppph(){
    // ###########################################################
    // ## Special care given to the vppph                       ##
    // ###########################################################
    //begin by assembling LHS
    uint Ndim = Np*(Np+1)*(Np+2)/6;
    ivec LHS(Ndim);
    uvec An(Ndim);
    uvec Bn(Ndim);
    uvec Cn(Ndim);

    uint ncount = 0;
    for(uint a = 0; a< Np; ++a){
        for(uint b= 0; b<=a; ++b){
            for(uint c= 0; c<=b; ++c){
                An(ncount) = a;
                Bn(ncount) = b;
                Cn(ncount) = c;
                ncount += 1;
            }
        }
    }
    LHS = eBs.unique(An + Nh) + eBs.unique(Bn+ Nh) - eBs.unique(Cn + Nh); //notice the minus
    ivec K_unique = unique(LHS);

    ivec RHS = eBs.unique(linspace<uvec>(0,Nh-1, Nh));
    K_unique = intersect1d(unique(RHS), K_unique); //the unique overlaps where preservation of quantum numbers occur

    uvec l_sorted = sort_index(LHS);
    //uvec l_sorted = linspace<uvec>(0,Ndim-1, Ndim);
    bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool br = false;
    bool row_collect = false;
    int ll_c = l_c;
    uint a;
    uint b;
    uint c;
    uint l_sorted_lc;
    uvec row_a(1000000);
    //uvec row_b(1000000);

    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(l_sorted_lc);

        if(l_c == C){
            a = An(l_sorted_lc);
            b = Bn(l_sorted_lc);
            c = Cn(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block

            row_a(nx) = a + b*Np + c*Np*Np;

            //row_b(nx) = b;
            if(a!=b){
                nx += 1;
                row_a(nx) = b + a*Np + c*Np*Np;

                //row_b(nx) = a;
            }
            //row(nx) = l_sorted(lc);
            nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                //uvec r = sort(row(span(0,nx-1)));
                tempRows(i) = sort(row_a(span(0,nx-1)));
                //tempRows(1)(i) = row_b(span(0,nx-1));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    tempRows(i) = sort(row_a(span(0,nx-1)));

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    //fmBlocks(uiCurrent_block).set_size(uiN);
    fmBlockz(uiCurrent_block).set_size(uiN,2);
    fuvCols(uiCurrent_block).set_size(uiN);
    fuvRows(uiCurrent_block).set_size(uiN);
    for(uint i = 0; i<uiN; ++i){
        uvec i2 = tempRows(i);
        uvec uvB = floor(i2/Np); //convert to unsigned integer indexing vector
        uvec uvA = i2 - uvB*Np;
        fmBlockz(uiCurrent_block)(i,0) = uvA; //reusing the framework here, naming does not matter
        fmBlockz(uiCurrent_block)(i,1) = uvB;
    }

    //lock and load
    uiCurrent_block += 1;
}


void blockmap::map_regions(imat L, imat R){

    fmOrdering(uiCurrent_block,0) = L;
    fmOrdering(uiCurrent_block,1) = R;



    // ###########################################################
    // ## Counting number of rows and columns in representation ##
    // ###########################################################
    uint iNrows = 1;
    uint iNcols = 1;

    for(uint i = 0; i<L.n_rows; ++i){
        iNrows *= uvSize(L(i,0));
    }
    for(uint i = 0; i<R.n_rows; ++i){
        iNcols *= uvSize(R(i,0));
    }

    // #########################################
    // ## Extract and organize actual indices ##
    // #########################################
    //uvec rows = conv_to<uvec>::from(linspace(0,iNrows-1, iNrows));

    uvec rows = linspace<uvec>(0,iNrows-1, iNrows);
    uvec cols = linspace<uvec>(0,iNcols-1, iNcols);

    //uvec cols = conv_to<uvec>::from(linspace(0,iNcols-1, iNcols));

    field<uvec> left = unpack(rows, L);
    field<uvec> right = unpack(cols, R);


    field<uvec> PQRS(4);
    for(uint i = 0; i< left.n_rows; ++i){
        PQRS(L(i,0)) = left(i);
    }
    for(uint i = 0; i< right.n_rows; ++i){
        PQRS(R(i,0)) = right(i);
    }
    //PQRS.print();

    // ############################################################
    // ## assign nonambiguous integer to each bra and ket config ##
    // ############################################################
    ivec LHS(iNrows); // = conv_to<ivec>::from(zeros(iNrows));
    for(uint i = 0; i < iNrows; ++i){
        LHS(i) = 0;
    }
    ivec RHS(iNcols); // = conv_to<ivec>::from(zeros(iNcols));
    for(uint i = 0; i < iNcols; ++i){
        RHS(i) = 0;
    }

    for(uint i = 0; i<L.n_rows; ++i){
        LHS += eBs.unique(PQRS(L(i,0))+L(i,2))*L(i,1);
    }
    for(uint i = 0; i<R.n_rows; ++i){
        RHS += eBs.unique(PQRS(R(i,0))+R(i,2))*R(i,1);
    }

    //LHS.print();

    // ####################################################################
    // ## Iterate over unique combinations, retain blocks where RHS==LHS ##
    // ####################################################################
    ivec unique_L = unique(LHS);
    ivec unique_R = unique(RHS);
    ivec K_unique = intersect1d(unique_L, unique_R);

    uint uiN = K_unique.n_elem;

    blocklengths(uiCurrent_block) = uiN; //number of blocks in config
    fvConfigs(uiCurrent_block) = K_unique; //ordering
    //fmBlocks(uiCurrent_block).set_size(uiN);
    fmBlockz(uiCurrent_block).set_size(uiN,2);
    fuvCols(uiCurrent_block).set_size(uiN);
    fuvRows(uiCurrent_block).set_size(uiN);


    field<uvec> tempElements(uiN);
    field<uvec> tempBlockmap1(uiN);
    field<uvec> tempBlockmap2(uiN);
    field<uvec> tempBlockmap3(uiN);

    //uint tempElementsSize = 0;

    field<uvec> tempRow = blocksort(LHS, K_unique);
    field<uvec> tempCol = blocksort(RHS, K_unique);


    for(uint i = 0; i<uiN; ++i){
        uvec row = rows.elem(find(LHS==K_unique(i)));
        uvec col = cols.elem(find(RHS==K_unique(i)));

        fmBlockz(uiCurrent_block)(i,0) = row;
        fmBlockz(uiCurrent_block)(i,1) = col;

        //fmBlockz(uiCurrent_block)(i,0) = tempRow(i);
        //fmBlockz(uiCurrent_block)(i,1) = tempCol(i);

        //fuvRows(uiCurrent_block)(i) = row;
        //fuvCols(uiCurrent_block)(i) = col;


    }

    //lock and load
    uiCurrent_block += 1;










} //map all regions defined by L == R

mat blockmap::getblock_vpppp(int u, int i){
    uvec a = fmBlockz(u)(i,0);
    uvec b = fmBlockz(u)(i,1);
    uint Nx = a.n_rows;
    //cout << Nx << " " << sqrt(Nx+1) << " " ;
    //a.print();
    //cout << endl;
    //b.print();
    //cout << endl << endl;
    mat block(Nx,Nx);
    double val;
    uint ax, bx, ay, by;
    for(uint nx = 0; nx<Nx; ++nx){
        ax = a(nx) +Nh;
        bx = b(nx) +Nh;
        // aaaa abab baba abba baab


        block(nx,nx) = eBs.v3(ax, bx, ax, bx);


        for(uint ny = nx; ny<Nx; ++ny){
            ay = a(ny) +Nh;
            by = b(ny) +Nh;
            val = eBs.v3(ax, bx, ay, by);
            block(nx,ny) = val;
            block(ny,nx) = val;
        }
    }

    return block;


}

mat blockmap::getblock(int u, int i){
    //umat block = fmBlocks(u)(i);
    //mat block = vElements.elem(fmBlocks(u)(i));
    //block.reshape(fmBlocks(u)(i).n_rows, fmBlocks(u)(i).n_cols);
    //return block;

    uvec row = fmBlockz(u)(i,0);
    uvec col = fmBlockz(u)(i,1);

    imat L = fmOrdering(u,0);
    imat R = fmOrdering(u,1);

    int Nx = row.n_rows;
    int Ny = col.n_rows;
    //cout << Nx << " " << Ny << " " << endl; //" " << K_unique(i) << endl;
    mat block(Nx,Ny);


    uvec pqrs(4);
    for(int nx = 0; nx < Nx; nx++){
        for(int ny = 0; ny < Ny; ny++){
            uvec lhs = unpack_uvec(row(nx), L);
            uvec rhs = unpack_uvec(col(ny), R);
            for(uint j = 0; j<lhs.n_elem; ++j){
                pqrs(L(j,0)) = lhs(j);
                if(uvSize(L(j,0))==Np){
                    pqrs(L(j,0)) += Nh;
                }
            }
            for(uint j = 0; j<rhs.n_elem; ++j){
                pqrs(R(j,0)) = rhs(j);
                if(uvSize(R(j,0))==Np){
                    pqrs(R(j,0)) += Nh;
                }
            }
            //index = to(pqrs(0), pqrs(1), pqrs(2), pqrs(3));
            block(nx, ny) = eBs.v3(pqrs(0), pqrs(1), pqrs(2), pqrs(3)); //remember to add in the needed shifts
        }
    }
    //cout << Nx << " " << Ny << " " << endl; //" " << K_unique(i) << endl;
    return block;

}

// ###########################################
// ##
// ##  SEQUENCES OF ROWS/COLUMNS
// ##
// ############################################

field<ivec> blockmap::pp(){
    //return a "compacted" particle particle unique indexvector

    //length of "compacted" vector
    uint N = Np*(Np+1)/2;

    //indices
    ivec a(N);
    ivec b(N);
    uint count = 0;
    for(int na = 0; na<Np; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            a(count) = na;
            b(count) = nb;
            count += 1;
        }
    }

    ivec Kab = eBs.unique(conv_to<uvec>::from(a) +Nh) + eBs.unique(conv_to<uvec>::from(b)+Nh);

    field<ivec> ppmap(3);
    ppmap(0) = a;
    ppmap(1) = b;
    ppmap(2) = Kab;
    return ppmap;
    }

field<ivec> blockmap::ppp(){
    //return a "compacted" particle particle particle unique indexvector
    //length of "compacted" vector
    uint N = Np*(Np+1)*(Np+2)/6;
    //indices
    ivec a(N);
    ivec b(N);
    ivec c(N);
    uint count = 0;
    for(int na = 0; na<Np; ++na){
        for(int nb = 0; nb<na+1; ++nb){
            for(int nc = 0; nc<nb+1; ++nc){
                a(count) = na;
                b(count) = nb;
                c(count) = nc;
                count += 1;
            }
        }
    }
    ivec Kabc = eBs.unique(conv_to<uvec>::from(a) +Nh) + eBs.unique(conv_to<uvec>::from(b)+Nh) + eBs.unique(conv_to<uvec>::from(c) +Nh);

    field<ivec> pppmap(4);
    pppmap(0) = a;
    pppmap(1) = b;
    pppmap(2) = c;
    pppmap(3) = Kabc;
    return pppmap;
    }

field<ivec> blockmap::hh(){
    //noncompacted
    ivec h = linspace<ivec>(0,Nh-1,Nh);
    field<ivec> hhmap(3);
    hhmap(0) = kron(ones<ivec>(Nh), h);
    hhmap(1) = kron(h, ones<ivec>(Nh));
    hhmap(2) = eBs.unique(conv_to<uvec>::from(hhmap(0))) + eBs.unique(conv_to<uvec>::from(hhmap(1)));
    return hhmap;
}


field<ivec> blockmap::hp(){
    ivec h = linspace<ivec>(0,Nh-1,Nh);
    ivec p = linspace<ivec>(0,Np-1,Np);
    field<ivec> hhmap(3);
    hhmap(0) = kron(h, ones<ivec>(Np));
    hhmap(1) = kron(ones<ivec>(Nh), p);
    hhmap(2) = eBs.unique(conv_to<uvec>::from(hhmap(0))) + eBs.unique(conv_to<uvec>::from(hhmap(1))+Nh);
    return hhmap;
}

field<ivec> blockmap::ph(){
    ivec h = linspace<ivec>(0,Nh-1,Nh);
    ivec p = linspace<ivec>(0,Np-1,Np);
    field<ivec> hhmap(3);
    hhmap(0) = kron(p, ones<ivec>(Nh));
    hhmap(1) = kron(ones<ivec>(Np), h);
    hhmap(2) = eBs.unique(conv_to<uvec>::from(hhmap(0))+Nh) + eBs.unique(conv_to<uvec>::from(hhmap(1)));
    return hhmap;
}

field<uvec> blockmap::partition(ivec LHS, ivec K_unique){
    //identify i blocks of LHS where LHS==K(i)
    //Note that by "blocks", we actually means indices corresponding to preserved quantum numbers
    uvec l_sorted = sort_index(LHS);
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);
    uvec row(1000000); //arbitrarily large number (biggest possible block)
    uint nx = 0;
    int l_c= LHS(l_sorted(lc));
    field<uvec> tempRows(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    bool row_collect = false;
    //int ll_c = l_c;

    while(lc < uiS){
        l_c = LHS(l_sorted(lc));

        //if element belongs to block
        if(l_c == C){
            //collect element
            row(nx) = l_sorted(lc);
            nx += 1;
            row_collect = true;
        }
        //if row is complete
        else{
            //if row contains elements
            if(row_collect){
                //collect block
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }


        lc += 1;
        //cout << l_c - C << endl;
    }


    return tempRows;

}


field<uvec> blockmap::partition_pp(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(2));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(2)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(2)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint a;
    uint b;
    //uint c;
    uint l_sorted_lc;
    uvec row(1000000);
    //uvec row_b(1000000);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));

    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(2)(l_sorted_lc);

        if(l_c == C){
            a = Na(l_sorted_lc);
            b = Nb(l_sorted_lc);
            //c = Cn(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block

            row(nx) = a + b*Np;

            //row_b(nx) = b;
            if(a!=b){
                nx += 1;
                row(nx) = b + a*Np;

                //row_b(nx) = a;
            }
            //row(nx) = l_sorted(lc);
            nx += 1;
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                //uvec r = sort(row(span(0,nx-1)));
                tempRows(i) = sort(row(span(0,nx-1)));
                //tempRows(1)(i) = row_b(span(0,nx-1));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    tempRows(i) = sort(row(span(0,nx-1)));
    return tempRows;
}

field<uvec> blockmap::partition_ppp(field<ivec> LHS, ivec K_unique){
    //partition particle-particle rows into blocks with preserved quantum numbers
    uvec l_sorted = sort_index(LHS(3));
    //bool adv = false;

    uint lc = 0;
    uint i = 0;
    uint uiN = K_unique.n_rows;
    uint uiS = l_sorted.n_rows;

    int C = K_unique(i);

    uint nx = 0;
    int l_c= LHS(3)(l_sorted(lc));
    field<uvec> tempRows(uiN);
    //tempRows(uiN) = K_unique;
    //tempRows(0).set_size(uiN);
    //tempRows(1).set_size(uiN);

    //align counters
    while(l_c<C){
        lc += 1;
        l_c = LHS(3)(l_sorted(lc));
    }

    //want to find row indices where LHS == k_config(i)
    //now: l_c == C
    //bool br = false;
    bool row_collect = false;
    //int ll_c = l_c;
    uint p,q,r;

    uint l_sorted_lc;
    uvec row(1000000);
    //uvec row_b(1000000);


    uvec Na = conv_to<uvec>::from(LHS(0));
    uvec Nb = conv_to<uvec>::from(LHS(1));
    uvec Nc = conv_to<uvec>::from(LHS(2));

    uint Np2 =Np*Np;
    while(lc < uiS){
        l_sorted_lc = l_sorted(lc);
        l_c = LHS(3)(l_sorted_lc);

        if(l_c == C){
            p = Na(l_sorted_lc);
            q = Nb(l_sorted_lc);
            r = Nc(l_sorted_lc);
            //the locations a + b*Np and b + a*Np (in the full array) are now identified to have LHS == C, belonging to the block
            row(nx) = p + q*Np + r*Np2;
            nx += 1;
            if(r!=q){
                row(nx) = p + r*Np + q*Np2;
                nx += 1;
                if(p!=q){
                    row(nx) = q + p*Np + r*Np2;
                    nx += 1;
                    row(nx) = q + r*Np + p*Np2;
                    nx += 1;
                    //it follows that p!=r, so
                    row(nx) = r + p*Np + q*Np2;
                    nx += 1;
                    row(nx) = r + q*Np + p*Np2;
                    nx += 1;
                }
            }
            else{
                if(p!=q){
                    row(nx) = q + p*Np + r*Np2;
                    nx += 1;
                }
                if(p!=r){
                    row(nx) = r + q*Np + p*Np2;
                    nx += 1;
                }
            }
            row_collect = true;
        }

        //if row is complete
        else{
            if(row_collect){
                tempRows(i) = sort(row(span(0,nx-1)));
                lc -= 1;
                i += 1;
                nx = 0;
                C = K_unique(i);
                row_collect = false;
            }
        }
        lc += 1;
    }

    //collect final block
    tempRows(i) = sort(row(span(0,nx-1)));
    return tempRows;
}
