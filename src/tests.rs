use super::*;
use crate::costs::sq_l2_dist;

type SeriesType = ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>;

#[test]
// Tests the naive DTW with different cost functions for sequences of EQUAL and UNEQUAL lengths
fn naive_dtw() {
    // ##### Sequences of UNEQUAL length #######
    let (query_unequal, data) = make_test_series(false);

    // UNEQUAL lengths + l2_dist
    let cost = naive::dtw(&data, &query_unequal, crate::costs::l2_dist, false);
    assert!((0.55 - cost).abs() < 0.000000000001);

    // UNEQUAL lengths + sq_l2_dist
    let cost = naive::dtw(&data, &query_unequal, sq_l2_dist, false);
    assert!((0.19329999999999 - cost).abs() < 0.000000000001);

    // ##### Sequences of EQUAL length #######
    let (query_equal, data) = make_test_series(true);
    // EQUAL lengths + l2_dist
    let cost = naive::dtw(&data, &query_equal, crate::costs::l2_dist, false);
    assert!((3.29 - cost).abs() < 0.000000000001);

    // EQUAL lengths + sq_l2_dist
    let cost = naive::dtw(&data, &query_equal, sq_l2_dist, false);
    assert!((4.5969 - cost).abs() < 0.000000000001);
}

#[test]
fn ucr_usp_dtw() {
    let (query, data) = make_test_series(true);
    // Creates dummy cummulative lower bound
    // We want the full calculation without abandoning or pruning so it consists only of 0.0
    let cb = vec![0.0; data.len()];
    let cost_ucr = ucr::dtw(
        &data,
        &query,
        &cb,
        data.len() - 2,
        f64::INFINITY,
        &sq_l2_dist,
    );
    assert!((4.5969 - cost_ucr).abs() < 0.000000000001);
}

#[test]
fn ucr_equals_naive_dtw() {
    let cost_fn = crate::costs::sq_l2_dist_1d;
    // Create random sequences for the query and the data time series
    // The observations are of type f64
    // The time series length is between 0 and 300
    let (data, query) = make_rdm_series((0, 300), None);
    // Creates dummy cummulative lower bound
    // We want the full calculation without abandoning or pruning so it consists only of 0.0
    let cb = vec![0.0; data.len()];
    let cost_ucr = ucr::dtw(&data, &query, &cb, data.len() - 2, f64::INFINITY, &cost_fn);
    let cost_naive = naive::dtw(&data, &query, cost_fn, false);
    assert!((cost_naive - cost_ucr).abs() < 0.000000000001);
}

// Create random sequences for the query and the data time series
// The observations are of type f64
// The tuple "query_rng" describes the min and max length of the query
// If a tuple "data_rng" was provided, it describes the min and max length of the data sequence
// Otherwise the data sequence has the same length as the query
fn make_rdm_series(
    query_rng: (u32, u32),
    data_rng: Option<(u32, u32)>,
) -> (std::vec::Vec<f64>, std::vec::Vec<f64>) {
    use rand::thread_rng;
    use rand::Rng;

    let mut rng = thread_rng();
    let query_len: u32 = rng.gen_range(query_rng.0..query_rng.1);
    let data_len: u32;
    if let Some((start, end)) = data_rng {
        data_len = rng.gen_range(start..end)
    } else {
        data_len = query_len;
    };

    let mut query = Vec::new();
    let mut data = Vec::new();

    for _ in 0..data_len {
        let observation: f64 = rand::random();
        data.push(observation);
    }
    for _ in 0..query_len {
        let observation: f64 = rand::random();
        query.push(observation);
    }

    (query, data)
}

fn make_test_series(equal_len: bool) -> (std::vec::Vec<SeriesType>, std::vec::Vec<SeriesType>) {
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
    let mut data = Vec::new();
    data.push(a1);
    data.push(a2);
    data.push(a3);
    data.push(a4);
    data.push(a5);
    data.push(a6);

    let mut query = Vec::new();
    query.push(b1);
    query.push(b2);
    query.push(b3);
    query.push(b4);
    query.push(b5);
    query.push(b6);

    if !equal_len {
        query.push(b7);
        query.push(b8);
    }
    (query, data)
}
