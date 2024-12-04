use abm::{abm_states::HifiStates, DoubleHifiProblemBuilder};
use faer::Mat;
use quantum::{cast_variant, params::{particle_factory::RotConst, particles::Particles}, states::{operator::Operator, state::State, state_type::StateType, States}, units::{energy_units::Energy, Au}};
use scattering_solver::potentials::{composite_potential::Composite, dispersion_potential::Dispersion, masked_potential::MaskedPotential, pair_potential::PairPotential, potential::{Potential, SimplePotential}};

use crate::utility::{percival_coef, RotorJMax, RotorJTot, RotorLMax};

#[derive(Clone, Copy, PartialEq)]
pub enum RotorAtomStates {
    SystemL,
    RotorJ,
}

#[derive(Clone)]
pub struct AlkaliDiatomAtomProblemBuilder<P, V>
where 
    P: SimplePotential,
    V: SimplePotential
{
    hifi_problem: DoubleHifiProblemBuilder,
    triplet_potential: Vec<(u32, P)>,
    singlet_potential: Vec<(u32, V)>,
}

impl<P, V> AlkaliDiatomAtomProblemBuilder<P, V> 
where 
    P: SimplePotential,
    V: SimplePotential
{
    pub fn new(hifi_problem: DoubleHifiProblemBuilder, triplet: Vec<(u32, P)>, singlet: Vec<(u32, V)>) -> Self {
        assert!(hifi_problem.first.s == 1);
        assert!(hifi_problem.second.s == 1);

        Self {
            hifi_problem,
            triplet_potential: triplet,
            singlet_potential: singlet,
        }
    }

    pub fn build(self, mag_field: f64, particles: &Particles) -> AlkaliDiatomAtomProblem<impl Potential<Space = Mat<f64>>> {
        let l_max = particles.get::<RotorLMax>().expect("Did not find SystemLMax parameter in particles").0;
        let j_max = particles.get::<RotorJMax>().expect("Did not find RotorJMax parameter in particles").0;
        let j_tot = particles.get::<RotorJTot>().map_or(0, |x| x.0);
        // todo! change to rotor particle having RotConst
        let rot_const = particles.get::<RotConst>().expect("Did not find RotConst parameter in the first particle").0;
        
        let all_even = self.triplet_potential.iter().all(|(lambda, _)| lambda & 1 == 0)
            && self.singlet_potential.iter().all(|(lambda, _)| lambda & 1 == 0);

        let ls = if all_even { (0..=l_max).step_by(2).collect() } else { (0..=l_max).collect() };
        let system_l = State::new(RotorAtomStates::SystemL, ls);
        let js = if all_even { (0..=j_max).step_by(2).collect() } else { (0..=j_max).collect() };
        let rotor_j = State::new(RotorAtomStates::RotorJ, js);
        
        let mut rotor_states = States::default();
        rotor_states.push_state(StateType::Irreducible(system_l))
            .push_state(StateType::Irreducible(rotor_j));

        let rotor_basis = rotor_states.get_basis();
        let hifi_problem = self.hifi_problem.build();
        let hifi_basis = hifi_problem.get_basis();

        let id_rotor = Mat::<f64>::identity(rotor_basis.len(), rotor_basis.len());
        let id_hifi = Mat::<f64>::identity(hifi_basis.len(), hifi_basis.len());
        
        let hifi_states = hifi_problem.states_at(mag_field);

        let l_centrifugal = Operator::from_diagonal_mel(&rotor_basis, [RotorAtomStates::SystemL], |[l]| {
            (l.1 * (l.1 + 1)) as f64
        });
        let l_centrifugal_mask = l_centrifugal.kron(&id_hifi);
        let l_potential = Dispersion::new(1. / 2. * particles.red_mass(), -2);

        let j_centrifugal = Operator::from_diagonal_mel(&rotor_basis, [RotorAtomStates::RotorJ], |[j]| {
            (j.1 * (j.1 + 1)) as f64
        });
        let j_centrifugal_mask = j_centrifugal.kron(&id_hifi);
        let j_potential = Dispersion::new(rot_const, 0);

        let hifi_masking = id_rotor.kron(Mat::from_fn(hifi_basis.len(), hifi_basis.len(), |i, j| {
            if i == j { hifi_states.0[i].to_au() } else { 0. }
        }));
        let hifi_potential = Dispersion::new(1., 0);

        let mut hifi_rotor_potential = Composite::new(MaskedPotential::new(hifi_potential, hifi_masking));
        hifi_rotor_potential.add_potential(MaskedPotential::new(l_potential, l_centrifugal_mask))
            .add_potential(MaskedPotential::new(j_potential, j_centrifugal_mask));

        let triplet_masking = Operator::from_diagonal_mel(hifi_basis, [HifiStates::ElectronDSpin(0)], |[e]| {
            let spin_e = cast_variant!(e.0, HifiStates::ElectronDSpin);

            if spin_e == 2 { 1. } else { 0. }
        });
        let triplet_hifi_masking = hifi_states.1.transpose() * triplet_masking.as_ref() * &hifi_states.1;

        let singlet_masking = Operator::from_diagonal_mel(hifi_basis, [HifiStates::ElectronDSpin(0)], |[e]| {
            let spin_e = cast_variant!(e.0, HifiStates::ElectronDSpin);

            if spin_e == 0 { 1. } else { 0. }
        });
        let singlet_hifi_masking = hifi_states.1.transpose() * singlet_masking.as_ref() * &hifi_states.1;

        let mut triplet_potentials = self.triplet_potential.into_iter()
            .map(|(lambda, potential)| {
                let rotor_masking = Operator::from_mel(&rotor_basis, [RotorAtomStates::SystemL, RotorAtomStates::RotorJ], |[l, j]| {
                    let lj_left = (l.bra.1, j.bra.1);
                    let lj_right = (l.ket.1, j.ket.1);

                    percival_coef(lambda, lj_left, lj_right, j_tot)
                });

                MaskedPotential::new(potential, rotor_masking.kron(&triplet_hifi_masking))
            });

        let mut triplet_potential = Composite::new(triplet_potentials.next().expect("No triplet potentials found"));
        for p in triplet_potentials {
            triplet_potential.add_potential(p);
        }

        let mut singlet_potentials = self.singlet_potential.into_iter()
            .map(|(lambda, potential)| {
                let rotor_masking = Operator::from_mel(&rotor_basis, [RotorAtomStates::SystemL, RotorAtomStates::RotorJ], |[l, j]| {
                    let lj_left = (l.bra.1, j.bra.1);
                    let lj_right = (l.ket.1, j.ket.1);

                    percival_coef(lambda, lj_left, lj_right, j_tot)
                });

                MaskedPotential::new(potential, rotor_masking.kron(&singlet_hifi_masking))
            });

        let mut singlet_potential = Composite::new(singlet_potentials.next().expect("No singlet potentials found"));
        for p in singlet_potentials {
            singlet_potential.add_potential(p);
        }

        let potential = PairPotential::new(triplet_potential, singlet_potential);
        let full_potential = PairPotential::new(hifi_rotor_potential, potential);

        AlkaliDiatomAtomProblem {
            potential: full_potential,
            channel_energies: hifi_states.0,
        }
    }
}

pub struct AlkaliDiatomAtomProblem<P: Potential<Space = Mat<f64>>> {
    pub potential: P,
    pub channel_energies: Vec<Energy<Au>>,
}
