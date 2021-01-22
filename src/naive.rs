use super::*;

/// Calculate the DTW cost of two sequences of vectors of n components with a naive implementation (no pruning or other optimizations)
pub fn dtw<F, T>(series_a: &[T], series_b: &[T], cost_fn: F, debug: bool) -> f64
where
    F: Fn(&T, &T) -> f64,
    T: std::fmt::Debug,
{
    // Init cost matrix
    let mut cost_mtrx =
        Array::<f64, Ix2>::from_elem((series_a.len() + 1, series_b.len() + 1), f64::INFINITY);
    cost_mtrx[[0, 0]] = 0.;

    // Calculate cost matrix
    for i in 1..=series_a.len() {
        for j in 1..=series_b.len() {
            let cost = cost_fn(&series_a[i - 1], &series_b[j - 1]);
            cost_mtrx[[i, j]] = cost
                + f64::min(
                    f64::min(cost_mtrx[[i - 1, j]], cost_mtrx[[i, j - 1]]),
                    cost_mtrx[[i - 1, j - 1]],
                );
        }
    }

    // Print cost matrix
    if debug {
        for row_no in 0..series_a.len() + 1 {
            for cell_no in 0..series_b.len() + 1 {
                print!(
                    "{:5} ",
                    (if cost_mtrx[[row_no, cell_no]] == f64::MAX {
                        String::from("-")
                    } else {
                        format!("{:.*}", 2, cost_mtrx[[row_no, cell_no]])
                    })
                );
            }
            println!();
        }
    }

    // Return final cost
    cost_mtrx[[series_a.len(), series_b.len()]]
}
