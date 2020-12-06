use core::ops::AddAssign;
use ndarray::prelude::*;
use ndarray::Data;
use ndarray_stats::DeviationExt;
use num_traits::{cast::ToPrimitive, sign::Signed};

// Calculate the L2 distance (euclidian distance) for a vector
pub fn l2_dist<T, A, D>(a: &ArrayBase<T, D>, b: &ArrayBase<T, D>) -> f64
where
    T: Data<Elem = A>,
    A: AddAssign + Clone + Signed + ToPrimitive,
    D: Dimension,
{
    a.l2_dist(b).unwrap()
}

/// Calculate the similarity of two sequences of vectors of n components with a naive implementation of DTW (no pruning or other optimizations)
/// Inspired by https://www.codesuji.com/2020/11/05/Rust-and-Dynamic-Time-Warping/
pub fn dtw_naive<F, T, A, D>(
    a: &[ArrayBase<T, D>],
    b: &[ArrayBase<T, D>],
    norm_func: F,
    debug: bool,
) -> f64
where
    F: Fn(&ArrayBase<T, D>, &ArrayBase<T, D>) -> f64,
    T: Data<Elem = A>,
    A: AddAssign + Clone + Signed + ToPrimitive,
    D: Dimension,
{
    // Init similarity matrix
    let a_len = a.len() + 1;
    let b_len = b.len() + 1;
    let mut similarity_mtrx = Array::<f64, Ix2>::from_elem((a_len, b_len), f64::MAX);
    similarity_mtrx[[0, 0]] = 0.;

    // Calculate similarity matrix
    for i in 1..a_len {
        for j in 1..b_len {
            let cost = norm_func(&a[i - 1], &b[j - 1]);
            similarity_mtrx[[i, j]] = cost
                + f64::min(
                    f64::min(similarity_mtrx[[i - 1, j]], similarity_mtrx[[i, j - 1]]),
                    similarity_mtrx[[i - 1, j - 1]],
                );
        }
    }

    // Print similarity matrix
    if debug {
        for row_no in 0..a_len {
            for cell_no in 0..b_len {
                print!(
                    "{:5} ",
                    (if similarity_mtrx[[row_no, cell_no]] == f64::MAX {
                        String::from("-")
                    } else {
                        format!("{:.*}", 2, similarity_mtrx[[row_no, cell_no]])
                    })
                );
            }
            println!();
        }
    }

    // Return final cost
    similarity_mtrx[[a_len - 1, b_len - 1]]
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
        let cost = dtw_naive(&series_1, &series_2, l2_dist, false);
        assert!((0.55 - cost).abs() < 0.000000000001);
    }
}
