use super::*;

/// Calculate the similarity of two sequences of vectors of n components with a naive implementation of DTW (no pruning or other optimizations)
/// Inspired by https://www.codesuji.com/2020/11/05/Rust-and-Dynamic-Time-Warping/
pub fn dtw<F, T, A, D>(
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
