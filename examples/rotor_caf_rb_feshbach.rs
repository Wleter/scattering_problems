use abm::{consts::Consts, get_hifi, get_zeeman_prop, utility::diagonalize};
use faer::Mat;
use quantum::{problem_selector::{get_args, ProblemSelector}, problems_impl, states::{operator::Operator, state::State, state_type::StateType, States, StatesBasis}, units::energy_units::{Energy, GHz}};

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb Feshbach",

);

impl Problems {
    fn get_states() {
        // let hifi_operator = get_hifi!(basis, RotorAtom::AtomS, RotorAtom::AtomI, hifi_atom.to_au());
        // let zeeman_prop = Operator::new(get_zeeman_prop!(basis, RotorAtom::AtomS, gamma_e).as_ref()
        //     + get_zeeman_prop!(basis, RotorAtom::RotorS, gamma_e).as_ref());

        // let zee_hifi: Mat<f64> = hifi_operator.as_ref() + mag_field * zeeman_prop.as_ref();
        // let (energies, u) = diagonalize(zee_hifi.as_ref());
    }
}