use abm::{consts::Consts, get_hifi, get_zeeman_prop, utility::diagonalize, DoubleHifiProblemBuilder};
use faer::Mat;
use quantum::{states::{operator::Operator, state::State, state_type::StateType, States, StatesBasis}, units::{energy_units::Energy, Au}};
use scattering_solver::potentials::{pair_potential::PairPotential, potential::SimplePotential};

use crate::alkali_atoms::{AlkaliPotential, HifiPotential};

#[derive(Clone, Copy, PartialEq)]
pub enum RotorAtomState {
    SystemL,
    RotorJ,
    RotorS(u32),
    RotorI(u32),
    AtomS(u32),
    AtomI(u32)
}

#[derive(Clone)]
pub struct AlkaliDiatomAtomProblemBuilder<P, V>
where 
    P: SimplePotential,
    V: SimplePotential
{
    hifi_problem: DoubleHifiProblemBuilder,
    triplet_potential: P,
    singlet_potential: V,
}

impl<P, V> AlkaliDiatomAtomProblemBuilder<P, V> 
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
            singlet_potential,
        }
    }

    pub fn build(self, mag_field: f64, l_max: u32, j_max: u32) -> AlkaliDiatomAtomProblem<P, V> {
        let s_rotor = self.hifi_problem.first.s;
        let i_rotor = self.hifi_problem.first.i;
        let s_atom = self.hifi_problem.second.s;
        let i_atom = self.hifi_problem.second.i;
        let m_tot = self.hifi_problem.total_projection.expect("for calculations use specific projection");

        assert!(s_rotor == 1);
        assert!(s_atom == 1);

        let ls = (0..=l_max as i32).collect();
        let system_l = State::new(RotorAtomState::SystemL, ls);
        let js = (0..=j_max as i32).collect();
        let rotor_j = State::new(RotorAtomState::RotorJ, js);
        
        let rotor_s = State::new(RotorAtomState::AtomS(1), vec![-1, 1]);
        let mi = (-(i_rotor as i32)..=(i_rotor as i32)).step_by(2).collect();
        let rotor_i = State::new(RotorAtomState::RotorI(i_rotor), mi);
    
        let atom_s = State::new(RotorAtomState::AtomS(1), vec![-1, 1]);
        let mi = (-(i_atom as i32)..=(i_atom as i32)).step_by(2).collect();
        let atom_i = State::new(RotorAtomState::AtomI(i_atom), mi);

        let gamma_e = -2.0 * Consts::BOHR_MAG;
        
        let mut states = States::default();
        states.push_state(StateType::Irreducible(system_l))
            .push_state(StateType::Irreducible(rotor_j))
            .push_state(StateType::Irreducible(rotor_s))
            .push_state(StateType::Irreducible(rotor_i))
            .push_state(StateType::Irreducible(atom_s))
            .push_state(StateType::Irreducible(atom_i));

        let basis: StatesBasis<_, _> = states.iter_elements()
            .filter(|s| {
                s.pairwise_iter()
                    .filter(|&(s, _)| !matches!(s, RotorAtomState::RotorJ | RotorAtomState::SystemL))
                    .map(|(_, m)| *m)
                    .sum::<i32>() == m_tot
                }
            )
            .collect();
        assert!(basis.is_empty(), "no states with total projection {m_tot}");

        todo!()
    }
}

pub struct AlkaliDiatomAtomProblem<P, V> 
where 
    P: SimplePotential,
    V: SimplePotential
{
    pub potential: PairPotential<HifiPotential, AlkaliPotential<P, V>>,
    pub channel_energies: Vec<Energy<Au>>,
}