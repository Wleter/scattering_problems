use std::time::Instant;

use abm::{DoubleHifiProblemBuilder, HifiProblemBuilder, Symmetry};
use faer::Mat;
use hhmmss::Hhmmss;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use num::complex::Complex64;
use quantum::{params::{particle_factory, particles::Particles}, problem_selector::{get_args, ProblemSelector}, problems_impl, units::{energy_units::{Energy, Kelvin, MHz}, Au, Unit}, utility::linspace};
use scattering_problems::alkali_atoms::{AlkaliAtomsProblem, AlkaliAtomsProblemBuilder};
use scattering_solver::{boundary::{Boundary, Direction}, numerovs::{multi_numerov::faer_backed::FaerRatioNumerov, propagator::MultiStepRule}, observables::s_matrix::HasSMatrix, potentials::{composite_potential::Composite, dispersion_potential::Dispersion, potential::Potential}, utility::save_data};

use rayon::prelude::*;

pub fn main() {
    Problems::select(&mut get_args());
}

pub struct Problems;

problems_impl!(Problems, "Li2 Feshbach",
    "potential values" => |_| Self::potential_values(),
    "feshbach resonance" => |_| Self::feshbach()
);

impl Problems {
    fn get_potential(projection: i32, mag_field: f64) -> AlkaliAtomsProblem<impl Potential<Space = Mat<f64>>> {
        let first = HifiProblemBuilder::new(1, 2)
            .with_hyperfine_coupling(Energy(228.2 / 1.5, MHz).to_au());

        let hifi_problem = DoubleHifiProblemBuilder::new_homo(first, Symmetry::Fermionic)
            .with_projection(projection);

        let mut li2_triplet = Composite::new(Dispersion::new(-1381., -6));
        li2_triplet.add_potential(Dispersion::new(2.19348e8, -12));

        let mut li2_singlet = Composite::new(Dispersion::new(-1381., -6));
        li2_singlet.add_potential(Dispersion::new(1.112e7, -12));

        AlkaliAtomsProblemBuilder::new(hifi_problem, li2_triplet, li2_singlet)
            .build(mag_field)
    }

    fn get_particles(energy: Energy<impl Unit>) -> Particles {
        let li_first = particle_factory::create_atom("Li6").unwrap();
        let li_second = particle_factory::create_atom("Li6").unwrap();

        Particles::new_pair(li_first, li_second, energy)
    }

    fn potential_values() {
        let alkali_problem = Self::get_potential(-4, 100.);
        let potential = &alkali_problem.potential;

        let mut potential_mat = Mat::<f64>::identity(potential.size(), potential.size());
        potential.value_inplace(1e4, &mut potential_mat);

        let distances = linspace(4., 200., 1000);

        let mut p1 = Vec::new();
        let mut p2 = Vec::new();
        let mut p12 = Vec::new();
        let mut p21 = Vec::new();
        for &distance in distances.iter().progress() {
            potential.value_inplace(distance, &mut potential_mat);
            p1.push(potential_mat[(0, 0)]);
            p2.push(potential_mat[(1, 1)]);
            p12.push(potential_mat[(0, 1)]);
            p21.push(potential_mat[(1, 0)]);
        }

        let header = "mag_field\tptoentials";
        let data = vec![distances, p1, p2, p12, p21];

        save_data("li2_potentials", header, &data)
            .unwrap()
    }

    fn feshbach() {
        ///////////////////////////////////

        let projection = 0;
        let channel = 0;
        let energy_relative = Energy(1e-7, Kelvin);

        let mut mag_fields = linspace(0., 620., 620);
        mag_fields.append(&mut linspace(620., 625., 500));
        mag_fields.append(&mut linspace(625., 1200., 575));


        ///////////////////////////////////

        let start = Instant::now();
        
        let scatterings = mag_fields.par_iter().progress().map(|&mag_field| {
            let alkali_problem = Self::get_potential(projection, mag_field);
            let energy = energy_relative.to_au() + alkali_problem.channel_energies[channel].to_au();

            let li2 = Self::get_particles(Energy(energy, Au));
            let potential = &alkali_problem.potential;

            let id = Mat::<f64>::identity(potential.size(), potential.size());
            let boundary = Boundary::new(4., Direction::Outwards, (1.001 * &id, 1.002 * &id));
            let step_rule = MultiStepRule::default();
            let mut numerov = FaerRatioNumerov::new(potential, &li2, step_rule, boundary);

            numerov.propagate_to(1.5e3);
            numerov.data.calculate_s_matrix(channel).get_scattering_length()
        })
        .collect::<Vec<Complex64>>();

        let elapsed = start.elapsed();
        println!("calculated in {}", elapsed.hhmmssxxx());

        let scatterings_re = scatterings.iter().map(|x| x.re).collect();
        let scatterings_im = scatterings.iter().map(|x| x.im).collect();

        let header = "mag_field\tscattering_re\tscattering_im";
        let data = vec![mag_fields, scatterings_re, scatterings_im];

        save_data("li2_scatterings", header, &data)
            .unwrap()
    }
}