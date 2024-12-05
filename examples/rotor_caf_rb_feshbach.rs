use core::f64;
use std::time::Instant;

use abm::{DoubleHifiProblemBuilder, HifiProblemBuilder};
use faer::Mat;
use hhmmss::Hhmmss;
use indicatif::ParallelProgressIterator;
use num::complex::Complex64;
use quantum::{params::{particle::Particle, particle_factory::{self, RotConst}, particles::Particles}, problem_selector::{get_args, ProblemSelector}, problems_impl, units::{energy_units::{Energy, GHz, Kelvin, MHz}, mass_units::{Dalton, Mass}, Au, Unit}, utility::linspace};
use scattering_problems::{alkali_atoms::{AlkaliAtomsProblem, AlkaliAtomsProblemBuilder}, alkali_diatom_atom::{AlkaliDiatomAtomProblem, AlkaliDiatomAtomProblemBuilder}, utility::{RotorJMax, RotorJTot, RotorLMax}};
use scattering_solver::{boundary::{Boundary, Direction}, numerovs::{multi_numerov::faer_backed::FaerRatioNumerov, propagator::MultiStepRule, single_numerov::SingleRatioNumerov}, observables::s_matrix::HasSMatrix, potentials::{composite_potential::Composite, dispersion_potential::Dispersion, potential::{Potential, SimplePotential}}, utility::save_data};

use rayon::prelude::*;

pub fn main() {
    Problems::select(&mut get_args());
}

struct Problems;

problems_impl!(Problems, "CaF + Rb Feshbach",
    "isotropic potential" => |_| Self::potentials(),
    "single channel isotropic scatterings" => |_| Self::single_chan_scatterings(),
    "isotropic feshbach" => |_| Self::feshbach_iso(),
    "rotor feshbach" => |_| Self::feshbach_rotor(),
    "rotor potentials" => |_| Self::rotor_potentials()

);

impl Problems {
    fn triplet_iso(config: usize) -> Composite<Dispersion> {
        let factors = [1.0286, 0.9717, 1.00268];

        let mut triplet = Composite::new(Dispersion::new(-3084., -6));
        triplet.add_potential(Dispersion::new(factors[config] * 2e9, -12));

        triplet
    }

    fn singlet_iso(config: usize) -> Composite<Dispersion> {
        let factors = [1.0196, 0.9815, 1.0037];

        let mut singlet = Composite::new(Dispersion::new(-3084., -6));
        singlet.add_potential(Dispersion::new(factors[config] * 5e8, -12));

        singlet
    }

    fn potential_aniso() -> Composite<Dispersion> {
        let mut singlet = Composite::new(Dispersion::new(-100., -6));
        singlet.add_potential(Dispersion::new(1e7, -12));

        singlet
    }

    fn get_potential_iso(config_triplet: usize, config_singlet: usize, projection: i32, mag_field: f64) -> AlkaliAtomsProblem<impl Potential<Space = Mat<f64>>> {
        let hifi_caf = HifiProblemBuilder::new(1, 1)
            .with_hyperfine_coupling(Energy(120., MHz).to_au());

        let hifi_rb = HifiProblemBuilder::new(1, 3)
            .with_hyperfine_coupling(Energy(6.83 / 2., GHz).to_au());

        let hifi_problem = DoubleHifiProblemBuilder::new(hifi_caf, hifi_rb).with_projection(projection);

        let triplet = Self::triplet_iso(config_triplet);
        let singlet = Self::singlet_iso(config_singlet);

        AlkaliAtomsProblemBuilder::new(hifi_problem, triplet, singlet)
            .build(mag_field)
    }

    fn get_particles(energy: Energy<impl Unit>) -> Particles {
        let caf = Particle::new("CaF", Mass(39.962590850 + 18.998403162, Dalton));
        let rb = particle_factory::create_atom("Rb87").unwrap();

        let mut particles = Particles::new_pair(caf, rb, energy);
        particles.insert(RotorLMax(2));
        particles.insert(RotorJMax(2));
        particles.insert(RotorJTot(0));
        particles.insert(RotConst(Energy(10.3, GHz).to_au()));

        particles
    }

    fn potentials() {
        let particles = Self::get_particles(Energy(1e-7, Kelvin));
        let triplet = Self::triplet_iso(0);
        let singlet = Self::singlet_iso(0);
        let aniso = Self::potential_aniso();

        let distances = linspace(4., 20., 200);
        let triplet_values = distances.iter().map(|&x| triplet.value(x)).collect();
        let singlet_values = distances.iter().map(|&x| singlet.value(x)).collect();
        let aniso_values = distances.iter().map(|&x| aniso.value(x)).collect();

        for config in [0, 1, 2] {
            println!("{config}");
            let triplet = Self::triplet_iso(config);
            let singlet = Self::singlet_iso(config);

            let boundary = Boundary::new(8.5, Direction::Outwards, (1.01, 1.02));
            let mut numerov = SingleRatioNumerov::new(&triplet, &particles, MultiStepRule::default(), boundary);
            numerov.propagate_to(1e4);
            println!("{:.2}", numerov.data.calculate_s_matrix().unwrap().get_scattering_length().re);
    
            let boundary = Boundary::new(7.2, Direction::Outwards, (1.01, 1.02));
            let mut numerov = SingleRatioNumerov::new(&singlet, &particles, MultiStepRule::default(), boundary);
            numerov.propagate_to(1e4);
            println!("{:.2}", numerov.data.calculate_s_matrix().unwrap().get_scattering_length().re);
        }

        let data = vec![distances, triplet_values, singlet_values, aniso_values];
        save_data("CaF_Rb_iso", "distance\ttriplet\tsinglet\taniso", &data)
            .unwrap();
    }

    fn single_chan_scatterings() {
        let particles = Self::get_particles(Energy(1e-7, Kelvin));

        let factors = linspace(0.95, 1.05, 500);

        let scatterings_triplet = factors.iter()
            .map(|x| {
                let mut triplet = Composite::new(Dispersion::new(-3084., -6));
                triplet.add_potential(Dispersion::new(x * 2e9, -12));

                let boundary = Boundary::new(8.5, Direction::Outwards, (1.01, 1.02));
                let mut numerov = SingleRatioNumerov::new(&triplet, &particles, MultiStepRule::default(), boundary);

                numerov.propagate_to(1e4);
                numerov.data.calculate_s_matrix().unwrap().get_scattering_length().re
            })
            .collect();

        let scatterings_singlet = factors.iter()
            .map(|x| {
                let mut singlet = Composite::new(Dispersion::new(-3084., -6));
                singlet.add_potential(Dispersion::new(x * 5e8, -12));

                let boundary = Boundary::new(7.2, Direction::Outwards, (1.01, 1.02));
                let mut numerov = SingleRatioNumerov::new(&singlet, &particles, MultiStepRule::default(), boundary);

                numerov.propagate_to(1e4);
                numerov.data.calculate_s_matrix().unwrap().get_scattering_length().re
            })
            .collect();

        let data = vec![factors, scatterings_triplet, scatterings_singlet];

        save_data("CaF_Rb_1chan_scatterings", "factors\ttriplet\tsinglet", &data)
            .unwrap();
    }

    fn feshbach_iso() {
        ///////////////////////////////////

        let projection = 2;
        let channel = 0;

        let config_triplet = 0;
        let config_singlet = 2;

        let energy_relative = Energy(1e-7, Kelvin);
        let mag_fields = linspace(0., 1000., 4000);
        
        linspace(0., 1000., 1000);

        ///////////////////////////////////

        let start = Instant::now();
        
        let scatterings = mag_fields.par_iter().progress().map(|&mag_field| {
            let alkali_problem = Self::get_potential_iso(config_triplet, config_singlet, projection, mag_field);
            let energy = energy_relative.to_au() + alkali_problem.channel_energies[channel].to_au();

            let caf_rb = Self::get_particles(Energy(energy, Au));
            let potential = &alkali_problem.potential;

            let id = Mat::<f64>::identity(potential.size(), potential.size());
            let boundary = Boundary::new(7.2, Direction::Outwards, (1.001 * &id, 1.002 * &id));
            let step_rule = MultiStepRule::default();
            let mut numerov = FaerRatioNumerov::new(potential, &caf_rb, step_rule, boundary);

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

        save_data(&format!("CaF_Rb_iso_scatterings_{config_triplet}_{config_singlet}"), header, &data)
            .unwrap()
    }

    fn get_potential(config_triplet: usize, config_singlet: usize, projection: i32, mag_field: f64, particles: &Particles) -> AlkaliDiatomAtomProblem<impl Potential<Space = Mat<f64>>> {
        let hifi_caf = HifiProblemBuilder::new(1, 1)
            .with_hyperfine_coupling(Energy(120., MHz).to_au());

        let hifi_rb = HifiProblemBuilder::new(1, 3)
            .with_hyperfine_coupling(Energy(6.83 / 2., GHz).to_au());

        let hifi_problem = DoubleHifiProblemBuilder::new(hifi_caf, hifi_rb).with_projection(projection);

        let triplet = Self::triplet_iso(config_triplet);
        let singlet = Self::singlet_iso(config_singlet);
        let aniso = Self::potential_aniso();

        let triplets = vec![(0, triplet), (2, aniso.clone())];
        let singlets = vec![(0, singlet), (2, aniso)];

        AlkaliDiatomAtomProblemBuilder::new(hifi_problem, triplets, singlets)
            .build(mag_field, particles)
    }

    fn feshbach_rotor() {
        ///////////////////////////////////

        let projection = 2;
        let channel = 0;

        let config_triplet = 0;
        let config_singlet = 0;

        let energy_relative = Energy(1e-7, Kelvin);
        let mag_fields = linspace(0., 1000., 4000);

        ///////////////////////////////////

        let start = Instant::now();
        
        let scatterings = mag_fields.par_iter().progress().map(|&mag_field| {
            let mut caf_rb = Self::get_particles(energy_relative);
            let alkali_problem = Self::get_potential(config_triplet, config_singlet, projection, mag_field, &caf_rb);

            let energy = energy_relative.to_au() + alkali_problem.channel_energies[channel].to_au();

            caf_rb.insert(Energy(energy, Au));
            let potential = &alkali_problem.potential;

            let id = Mat::<f64>::identity(potential.size(), potential.size());
            let boundary = Boundary::new(8.5, Direction::Outwards, (1.001 * &id, 1.002 * &id));
            let step_rule = MultiStepRule::new(1e-3, f64::INFINITY, 500.);
            let mut numerov = FaerRatioNumerov::new(potential, &caf_rb, step_rule, boundary);

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

        save_data(&format!("CaF_Rb_scatterings_{config_triplet}_{config_singlet}"), header, &data)
            .unwrap()
    }

    fn rotor_potentials() {
        ///////////////////////////////////

        let projection = 4;
        let channel = 0;

        let config_triplet = 0;
        let config_singlet = 0;

        let energy_relative = Energy(1e-7, Kelvin);
        let distances = linspace(4.2, 30., 200);

        ///////////////////////////////////

        let mut caf_rb = Self::get_particles(energy_relative);
        let alkali_problem = Self::get_potential(config_triplet, config_singlet, projection, 100., &caf_rb);

        let energy = energy_relative.to_au() + alkali_problem.channel_energies[channel].to_au();

        caf_rb.insert(Energy(energy, Au));
        let potential = &alkali_problem.potential;
        
        let mut mat = Mat::zeros(potential.size(), potential.size());
        let potentials: Vec<Mat<f64>> = distances.iter().map(|&x| {
                potential.value_inplace(x, &mut mat);

                mat.to_owned()
            })
            .collect();
        let header = "distances\tpotentials";
        let mut data = vec![distances];
        for i in 0..potential.size() {
            for j in 0..potential.size() {
                data.push(potentials.iter().map(|p| p[(i, j)]).collect());
            }
        }

        save_data(&format!("CaF_Rb_potentials_{config_triplet}_{config_singlet}"), header, &data)
            .unwrap()
    }
}