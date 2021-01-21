use super::*;

/// Calculate the similarity of two sequences of vectors of n components with a naive implementation of DTW (no pruning or other optimizations)
pub fn dtw<F, T>(a: &[T], b: &[T], cost_fn: F, debug: bool) -> f64
where
    F: Fn(&T, &T) -> f64,
{
    // Init similarity matrix
    let mut similarity_mtrx = Array::<f64, Ix2>::from_elem((a.len() + 1, b.len() + 1), f64::MAX);
    similarity_mtrx[[0, 0]] = 0.;

    // Calculate similarity matrix
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = cost_fn(&a[i - 1], &b[j - 1]);
            similarity_mtrx[[i, j]] = cost
                + f64::min(
                    f64::min(similarity_mtrx[[i - 1, j]], similarity_mtrx[[i, j - 1]]),
                    similarity_mtrx[[i - 1, j - 1]],
                );
        }
    }

    // Print similarity matrix
    if debug {
        for row_no in 0..a.len() + 1 {
            for cell_no in 0..b.len() + 1 {
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
    similarity_mtrx[[a.len(), b.len()]]
}
