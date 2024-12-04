use abm::{consts::Consts, get_hifi, get_zeeman_prop, utility::diagonalize};
use faer::Mat;
use quantum::{params::particles::Particles, problem_selector::{get_args, ProblemSelector}, problems_impl, states::{operator::Operator, state::State, state_type::StateType, States, StatesBasis}, units::energy_units::{Energy, GHz}};

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb Feshbach",

);

impl Problems {
    fn get_states() {
        
    }

    fn get_particles() -> Particles {
        
    }
}