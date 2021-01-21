use core::ops::AddAssign;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray_stats::DeviationExt;
use num_traits::{cast::ToPrimitive, sign::Signed};
use std::fs::File;
use std::io::prelude::*;

pub mod naive;
mod tests;
pub mod ucr;

/// Calculate the squared L2 distance (euclidian distance) between two vectors
/// Uses the sq_l2_dist method of the ndarray crate
/// Read its documentation for more details
pub fn sq_l2_dist<T, A, D>(a: &ArrayBase<T, D>, b: &ArrayBase<T, D>) -> f64
where
    T: Data<Elem = A>,
    A: AddAssign + Clone + Signed + ToPrimitive,
    D: Dimension,
{
    a.sq_l2_dist(b)
        .unwrap()
        .to_f64()
        .expect("failed cast from type A to f64")
}

/// Calculate the squared L2 distance (euclidian distance) between two vectors
/// Uses the sq_l2_dist method of the ndarray crate
/// Read its documentation for more details
pub fn l2_dist<T, A, D>(a: &ArrayBase<T, D>, b: &ArrayBase<T, D>) -> f64
where
    T: Data<Elem = A>,
    A: AddAssign + Clone + Signed + ToPrimitive,
    D: Dimension,
{
    f64::sqrt(sq_l2_dist(a, b))
}

pub mod utilities {
    use super::*;

    pub struct DataContainer {
        pub data: Vec<f64>,
        index: usize,
    }

    impl DataContainer {
        pub fn new(filename: &str) -> Self {
            // Open the file in read-only mode.
            let mut file = File::open(filename).unwrap();
            let mut content = String::new();
            // Read all the file content into a variable (ignoring the result of the operation).
            file.read_to_string(&mut content).unwrap();
            let data = content
                .split_whitespace()
                .filter_map(|w| w.parse::<f64>().ok())
                .collect();
            DataContainer { data, index: 0 }
        }
    }

    impl Iterator for DataContainer {
        type Item = f64;
        fn next(&mut self) -> Option<f64> {
            let result = self.data.get(self.index);
            self.index += 1;
            result.cloned()
        }
    }

    struct DataReader<T: Iterator> {
        data: T,
    }

    impl<T, I> Iterator for DataReader<T>
    where
        T: Iterator<Item = I>,
    {
        type Item = I;
        fn next(&mut self) -> Option<I> {
            self.data.next()
        }
    }

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
