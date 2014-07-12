#include "Hamiltonian.h"

#define j1 couplingConstants[0]
#define j2 couplingConstants[1]
#define sigmaplus siteBasisH2[0]
#define sigmaz siteBasisH2[1]
#define sigmaminus siteBasisH2[2]
#define rhoBasisSigmaplus rhoBasisH2[0]
#define rhoBasisSigmaz rhoBasisH2[1]
#define off0RhoBasisSigmaplus off0RhoBasisH2[0]
#define off0RhoBasisSigmaz off0RhoBasisH2[1]
#define off1RhoBasisSigmaplus off1RhoBasisH2[0]
#define off1RhoBasisSigmaz off1RhoBasisH2[1]

using namespace Eigen;

Hamiltonian::Hamiltonian() : oneSiteQNums({1, -1})
{
    siteBasisH2.resize(3);
    sigmaplus << 0., 1.,
                 0., 0.;
    sigmaminus << 0., 0.,
                  1., 0.;
    sigmaz << 1.,  0.,
              0., -1.;                                 // define Pauli matrices
};

void Hamiltonian::setParams(const std::vector<double>& couplingConstantsIn,
                            int targetQNumIn, int lSysIn)
{
    couplingConstants = couplingConstantsIn;
    targetQNum = targetQNumIn;
    lSys = lSysIn;
};

MatrixX_t Hamiltonian::blockAdjacentSiteJoin(int j,
                                             const std::vector<MatrixX_t>& rhoBasisH2)
                                             const
{
    MatrixX_t plusMinus = kp(rhoBasisSigmaplus, sigmaminus);
    return couplingConstants[j - 1] *
        (kp(rhoBasisSigmaz, sigmaz) + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::lBlockrSiteJoin(const std::vector<MatrixX_t>&
                                       off0RhoBasisH2, int compm) const
{
    MatrixX_t plusMinus = kp(kp(off0RhoBasisSigmaplus, Id(d * compm)),
                             sigmaminus);
    return j2 * (kp(kp(off0RhoBasisSigmaz, Id(d * compm)), sigmaz)
                 + 2 * (plusMinus + plusMinus.adjoint()));
};

MatrixX_t Hamiltonian::lSiterBlockJoin(int m,
                                       const std::vector<MatrixX_t>&
                                       off0RhoBasisH2) const
{
    MatrixX_t plusMinus = kp(sigmaplus, off0RhoBasisSigmaplus.adjoint());
    return j2 *
        kp(kp(Id(m), kp(sigmaz, off0RhoBasisSigmaz)
                     + 2 * (plusMinus + plusMinus.adjoint())),
           Id_d);
};

MatrixX_t Hamiltonian::siteSiteJoin(int m, int compm) const
{
    MatrixX_t plusMinus = kp(kp(sigmaplus, Id(compm)), sigmaminus);
    return j1 * kp(Id(m), kp(kp(sigmaz, Id(compm)), sigmaz)
                           + 2 * (plusMinus + plusMinus.adjoint()));
};
