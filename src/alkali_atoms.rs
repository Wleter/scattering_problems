use abm::{abm_states::HifiStates, DoubleHifiProblemBuilder};
use faer::Mat;
use quantum::{cast_variant, states::operator::Operator, units::{energy_units::Energy, Au}};
use scattering_solver::potentials::{dispersion_potential::Dispersion, masked_potential::MaskedPotential, multi_diag_potential::Diagonal, pair_potential::PairPotential, potential::SimplePotential};

pub type AlkaliPotential<P, V> = PairPotential<MaskedPotential<Mat<f64>, P>, MaskedPotential<Mat<f64>, V>>;
pub type HifiPotential = Diagonal<Mat<f64>, Dispersion>;

#[derive(Clone)]
pub struct AlkaliAtomsProblemBuilder<P, V>
where 
    P: SimplePotential,
    V: SimplePotential
{
    hifi_problem: DoubleHifiProblemBuilder,
    triplet_potential: P,
    singlet_potential: V,
}

impl<P, V> AlkaliAtomsProblemBuilder<P, V> 
where 
    P: SimplePotential,
    V: SimplePotential
{
    pub fn new(hifi_problem: DoubleHifiProblemBuilder, triplet_potential: P, singlet_potential: V) -> Self {
        assert!(hifi_problem.first.s == 1);
        assert!(hifi_problem.second.s == 1);

        Self {
            hifi_problem,
            triplet_potential,
            singlet_potential
        }
    }

    pub fn build(self, magnetic_field: f64) -> AlkaliAtomsProblem<P, V> {
        let hifi = self.hifi_problem.build();
        let basis = hifi.get_basis();
        
        let hifi_states = hifi.states_at(magnetic_field);

        let triplet_masking = Operator::from_diagonal_mel(basis, [HifiStates::ElectronDSpin(0)], |[e]| {
            let spin_e = cast_variant!(e.0, HifiStates::ElectronDSpin);

            if spin_e == 2 { 1. } else { 0. }
        });
        let triplet_masking = hifi_states.1.transpose() * triplet_masking.as_ref() * &hifi_states.1;
        let triplet_potential = MaskedPotential::new(self.triplet_potential, triplet_masking);

        let singlet_masking = Operator::from_diagonal_mel(basis, [HifiStates::ElectronDSpin(0)], |[e]| {
            let spin_e = cast_variant!(e.0, HifiStates::ElectronDSpin);

            if spin_e == 0 { 1. } else { 0. }
        });
        let singlet_masking = hifi_states.1.transpose() * singlet_masking.as_ref() * &hifi_states.1;
        let singlet_potential = MaskedPotential::new(self.singlet_potential, singlet_masking);

        let hifi_potential = hifi_states.0.iter()
            .map(|e| Dispersion::new(e.to_au(), 0))
            .collect();
        let hifi_potential = HifiPotential::from_vec(hifi_potential);

        let potential = PairPotential::new(triplet_potential, singlet_potential);
        let full_potential = PairPotential::new(hifi_potential, potential);

        AlkaliAtomsProblem {
            potential: full_potential,
            channel_energies: hifi_states.0,
        }
    }
}

pub struct AlkaliAtomsProblem<P, V> 
where 
    P: SimplePotential,
    V: SimplePotential
{
    pub potential: PairPotential<HifiPotential, AlkaliPotential<P, V>>,
    pub channel_energies: Vec<Energy<Au>>,
}
