use clebsch_gordan::{wigner_3j, wigner_6j};

pub fn percival_coef(lambda: u32, lj_left: (u32, u32), lj_right: (u32, u32), j_tot: u32) -> f64 {
    let l_left = lj_left.0;
    let j_left = lj_left.1;

    let l_right = lj_right.0;
    let j_right = lj_right.1;

    let mut wigners = wigner_3j(2 * l_left, 2 * lambda, 2 * l_right, 0, 0, 0);
    if wigners == 0. { return 0. }

    wigners *= wigner_3j(2 * j_left, 2 * lambda, 2 * j_right, 0, 0, 0);
    if wigners == 0. { return 0. }

    wigners *= wigner_6j(l_left, lambda, l_right, j_right, j_tot, j_left);
    if wigners == 0. { return 0. }
    
    let sign = (-1.0f64).powi((l_left + l_right) as i32 - j_tot as i32);
    let prefactor = (((2 * j_left + 1) * (2 * j_right + 1) 
                        * (2 * l_left + 1) * (2 * l_right + 1)) as f64).sqrt();

    sign * prefactor * wigners
}

#[derive(Clone, Copy, Debug)]
pub struct RotorLMax(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct RotorJMax(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct RotorJTot(pub u32);
