/// Calculate the DTW cost of two sequences of vectors of n components with a naive implementation (no pruning or other optimizations)
/// A Sakoe-Chiba band is used to restrict the warping
pub fn dtw<F, T>(series_a: &[T], series_b: &[T], w: usize, cost_fn: F) -> f64
where
    F: Fn(&T, &T) -> f64,
    T: std::fmt::Debug,
{
    // Init cost matrix
    let row = vec![f64::INFINITY; series_a.len() + 1];
    let mut cost_mtrx = vec![row; series_b.len() + 1];
    cost_mtrx[0][0] = 0.;

    // Calculate cost matrix
    for i in 1..=series_b.len() {
        let w_start = (i - 1).saturating_sub(w) + 1;
        let w_end = usize::min(series_a.len(), i + w);

        for j in w_start..=w_end {
            let cost = cost_fn(&series_a[j - 1], &series_b[i - 1]);
            cost_mtrx[i][j] = cost
                + f64::min(
                    f64::min(cost_mtrx[i - 1][j], cost_mtrx[i][j - 1]),
                    cost_mtrx[i - 1][j - 1],
                );
        }
    }

    // Print cost matrix
    /*
        for row_no in 0..series_a.len() + 1 {
            for cell_no in 0..series_b.len() + 1 {
                print!(
                    "{:5} ",
                    (if cost_mtrx[row_no][cell_no] == f64::MAX {
                        String::from("-")
                    } else {
                        format!("{:.*}", 2, cost_mtrx[row_no][cell_no])
                    })
                );
            }
            println!();
        }
    }*/

    // Return final cost
    cost_mtrx[series_b.len()][series_a.len()]
}
