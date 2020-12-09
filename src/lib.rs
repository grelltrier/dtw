#[macro_use]
extern crate log;

use core::ops::AddAssign;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray_stats::DeviationExt;
use num_traits::{cast::ToPrimitive, sign::Signed};

pub mod naive;
pub mod ucr;

// Calculate the L2 distance (euclidian distance) for a vector
pub fn l2_dist<T, A, D>(a: &ArrayBase<T, D>, b: &ArrayBase<T, D>) -> f64
where
    T: Data<Elem = A>,
    A: AddAssign + Clone + Signed + ToPrimitive,
    D: Dimension,
{
    a.l2_dist(b).unwrap()
}

pub mod utilities {
    use super::*;
    type SeriesType = ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;
    pub fn make_test_series() -> ([SeriesType; 6], [SeriesType; 8]) {
        let a1 = array![1.0, 1.];
        let a2 = array![2.0, 1.];
        let a3 = array![3.0, 1.];
        let a4 = array![2.0, 1.];
        let a5 = array![2.13, 1.];
        let a6 = array![1.0, 1.];
        let b1 = array![1.0, 1.];
        let b2 = array![1.0, 1.];
        let b3 = array![2.0, 1.];
        let b4 = array![2.0, 1.];
        let b5 = array![2.42, 1.];
        let b6 = array![3.0, 1.];
        let b7 = array![2.0, 1.];
        let b8 = array![1.0, 1.];
        let series_1 = [a1, a2, a3, a4, a5, a6];
        let series_2 = [b1, b2, b3, b4, b5, b6, b7, b8];
        (series_1, series_2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cost_function() {
        let (series_1, series_2) = utilities::make_test_series();
        let cost = naive::dtw(&series_1, &series_2, l2_dist, false);
        assert!((0.55 - cost).abs() < 0.000000000001);
    }
}
