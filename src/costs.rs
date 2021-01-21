use super::*;

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

// Calculate the L2 distance (euclidian distance) for f64
pub fn sq_l2_dist_1d(x: &f64, y: &f64) -> f64 {
    (x - y).powi(2)
}
