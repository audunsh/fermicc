#ifndef FLEXMAT_H
#define FLEXMAT_H
#define ARMA_64BIT_WORD
#include <armadillo>
#include <iomanip>
#include "solver/unpack_sp_mat.h"

using namespace std;
using namespace arma;

class flexmat
{
public:
    //A class for flexible matrix manipulations
    flexmat();

    int iNp;
    int iNq;
    int iNr;
    int iNs;

    int iNp2;
    int iNh;
    int iNh2;
    int iNhp;
    int iNh2p;
    int iNp2h;

    void init(vec values, uvec p, uvec q, uvec r, uvec s, int Np, int Nq, int Nr, int Ns);

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

    mat rows_dense_mp(uvec urows, ivec &mcols);
    double intensity();

    void shed_zeros();

    void set_amplitudes(vec Energy);

    void partition(field<vec> fBlocks);

    void report();

    void update(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    vec vEnergy;

    //electrongas eBs;


    vec vValues;
    uvec vp;
    uvec vq;
    uvec vr;
    uvec vs;

    sp_mat smV;
    sp_mat rows(uvec urows); //returns a identically sized sp_mat with only urows set to non-zero
    mat rows_dense(uvec urows); //returns a identically sized sp_mat with only urows set to non-zero

    umat locations;

    void deinit();
    //update script, not for human maintenance
    void update_as_p_qrs(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_qrs_p(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_p_qsr(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_qsr_p(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_p_rqs(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_rqs_p(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_p_rsq(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_rsq_p(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_p_sqr(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_sqr_p(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_p_srq(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_srq_p(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_q_prs(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_prs_q(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_q_psr(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_psr_q(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_q_rps(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_rps_q(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_q_rsp(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_rsp_q(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_q_spr(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_spr_q(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_q_srp(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_srp_q(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_r_pqs(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_pqs_r(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_r_psq(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_psq_r(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_r_qps(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_qps_r(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_r_qsp(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_qsp_r(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_r_spq(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_spq_r(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_r_sqp(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_sqp_r(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_s_pqr(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_pqr_s(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_s_prq(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_prq_s(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_s_qpr(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_qpr_s(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_s_qrp(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_qrp_s(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_s_rpq(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_rpq_s(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    void update_as_s_rqp(sp_mat spC, int Np, int Nq, int Nr, int Ns);
    void update_as_rqp_s(sp_mat spC, int Np, int Nq, int Nr, int Ns);

    //initialization budget (generated in separate script - not for human maintenance)
    int Npq_rs;
    sp_mat Vpq_rs;
    sp_mat pq_rs();
    int Npq_sr;
    sp_mat Vpq_sr;
    sp_mat pq_sr();
    int Npr_qs;
    sp_mat Vpr_qs;
    sp_mat pr_qs();
    int Npr_sq;
    sp_mat Vpr_sq;
    sp_mat pr_sq();
    int Nps_qr;
    sp_mat Vps_qr;
    sp_mat ps_qr();
    int Nps_rq;
    sp_mat Vps_rq;
    sp_mat ps_rq();
    int Nqp_rs;
    sp_mat Vqp_rs;
    sp_mat qp_rs();
    int Nqp_sr;
    sp_mat Vqp_sr;
    sp_mat qp_sr();
    int Nqr_ps;
    sp_mat Vqr_ps;
    sp_mat qr_ps();
    int Nqr_sp;
    sp_mat Vqr_sp;
    sp_mat qr_sp();
    int Nqs_pr;
    sp_mat Vqs_pr;
    sp_mat qs_pr();
    int Nqs_rp;
    sp_mat Vqs_rp;
    sp_mat qs_rp();
    int Nrp_qs;
    sp_mat Vrp_qs;
    sp_mat rp_qs();
    int Nrp_sq;
    sp_mat Vrp_sq;
    sp_mat rp_sq();
    int Nrq_ps;
    sp_mat Vrq_ps;
    sp_mat rq_ps();
    int Nrq_sp;
    sp_mat Vrq_sp;
    sp_mat rq_sp();
    int Nrs_pq;
    sp_mat Vrs_pq;
    sp_mat rs_pq();
    int Nrs_qp;
    sp_mat Vrs_qp;
    sp_mat rs_qp();
    int Nsp_qr;
    sp_mat Vsp_qr;
    sp_mat sp_qr();
    int Nsp_rq;
    sp_mat Vsp_rq;
    sp_mat sp_rq();
    int Nsq_pr;
    sp_mat Vsq_pr;
    sp_mat sq_pr();
    int Nsq_rp;
    sp_mat Vsq_rp;
    sp_mat sq_rp();
    int Nsr_pq;
    sp_mat Vsr_pq;
    sp_mat sr_pq();
    int Nsr_qp;
    sp_mat Vsr_qp;
    sp_mat sr_qp();
    int Np_qrs;
    sp_mat Vp_qrs;
    sp_mat p_qrs();
    int Np_qsr;
    sp_mat Vp_qsr;
    sp_mat p_qsr();
    int Np_rqs;
    sp_mat Vp_rqs;
    sp_mat p_rqs();
    int Np_rsq;
    sp_mat Vp_rsq;
    sp_mat p_rsq();
    int Np_sqr;
    sp_mat Vp_sqr;
    sp_mat p_sqr();
    int Np_srq;
    sp_mat Vp_srq;
    sp_mat p_srq();
    int Nq_prs;
    sp_mat Vq_prs;
    sp_mat q_prs();
    int Nq_psr;
    sp_mat Vq_psr;
    sp_mat q_psr();
    int Nq_rps;
    sp_mat Vq_rps;
    sp_mat q_rps();
    int Nq_rsp;
    sp_mat Vq_rsp;
    sp_mat q_rsp();
    int Nq_spr;
    sp_mat Vq_spr;
    sp_mat q_spr();
    int Nq_srp;
    sp_mat Vq_srp;
    sp_mat q_srp();
    int Nr_pqs;
    sp_mat Vr_pqs;
    sp_mat r_pqs();
    int Nr_psq;
    sp_mat Vr_psq;
    sp_mat r_psq();
    int Nr_qps;
    sp_mat Vr_qps;
    sp_mat r_qps();
    int Nr_qsp;
    sp_mat Vr_qsp;
    sp_mat r_qsp();
    int Nr_spq;
    sp_mat Vr_spq;
    sp_mat r_spq();
    int Nr_sqp;
    sp_mat Vr_sqp;
    sp_mat r_sqp();
    int Ns_pqr;
    sp_mat Vs_pqr;
    sp_mat s_pqr();
    int Ns_prq;
    sp_mat Vs_prq;
    sp_mat s_prq();
    int Ns_qpr;
    sp_mat Vs_qpr;
    sp_mat s_qpr();
    int Ns_qrp;
    sp_mat Vs_qrp;
    sp_mat s_qrp();
    int Ns_rpq;
    sp_mat Vs_rpq;
    sp_mat s_rpq();
    int Ns_rqp;
    sp_mat Vs_rqp;
    sp_mat s_rqp();
    int Npqr_s;
    sp_mat Vpqr_s;
    sp_mat pqr_s();
    int Npqs_r;
    sp_mat Vpqs_r;
    sp_mat pqs_r();
    int Nprq_s;
    sp_mat Vprq_s;
    sp_mat prq_s();
    int Nprs_q;
    sp_mat Vprs_q;
    sp_mat prs_q();
    int Npsq_r;
    sp_mat Vpsq_r;
    sp_mat psq_r();
    int Npsr_q;
    sp_mat Vpsr_q;
    sp_mat psr_q();
    int Nqpr_s;
    sp_mat Vqpr_s;
    sp_mat qpr_s();
    int Nqps_r;
    sp_mat Vqps_r;
    sp_mat qps_r();
    int Nqrp_s;
    sp_mat Vqrp_s;
    sp_mat qrp_s();
    int Nqrs_p;
    sp_mat Vqrs_p;
    sp_mat qrs_p();
    int Nqsp_r;
    sp_mat Vqsp_r;
    sp_mat qsp_r();
    int Nqsr_p;
    sp_mat Vqsr_p;
    sp_mat qsr_p();
    int Nrpq_s;
    sp_mat Vrpq_s;
    sp_mat rpq_s();
    int Nrps_q;
    sp_mat Vrps_q;
    sp_mat rps_q();
    int Nrqp_s;
    sp_mat Vrqp_s;
    sp_mat rqp_s();
    int Nrqs_p;
    sp_mat Vrqs_p;
    sp_mat rqs_p();
    int Nrsp_q;
    sp_mat Vrsp_q;
    sp_mat rsp_q();
    int Nrsq_p;
    sp_mat Vrsq_p;
    sp_mat rsq_p();
    int Nspq_r;
    sp_mat Vspq_r;
    sp_mat spq_r();
    int Nspr_q;
    sp_mat Vspr_q;
    sp_mat spr_q();
    int Nsqp_r;
    sp_mat Vsqp_r;
    sp_mat sqp_r();
    int Nsqr_p;
    sp_mat Vsqr_p;
    sp_mat sqr_p();
    int Nsrp_q;
    sp_mat Vsrp_q;
    sp_mat srp_q();
    int Nsrq_p;
    sp_mat Vsrq_p;
    sp_mat srq_p();






};

#endif // FLEXMAT_H
