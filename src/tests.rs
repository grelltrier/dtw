use super::*;

#[test]
// Tests the naive DTW with different cost functions for sequences of EQUAL and UNEQUAL lengths
fn naive_dtw() {
    // ##### Sequences of UNEQUAL length #######
    let (query_unequal, data) = make_test_series(false);

    // UNEQUAL lengths + l2_dist
    let cost = naive::dtw(&data, &query_unequal, dtw_cost::l2_dist_vec, false);
    assert!((0.55 - cost).abs() < 0.000000000001);

    // UNEQUAL lengths + sq_l2_dist
    let cost = naive::dtw(&data, &query_unequal, dtw_cost::sq_l2_dist_vec, false);
    assert!((0.19329999999999 - cost).abs() < 0.000000000001);

    // ##### Sequences of EQUAL length #######
    let (query_equal, data) = make_test_series(true);
    // EQUAL lengths + l2_dist
    let cost = naive::dtw(&data, &query_equal, dtw_cost::l2_dist_vec, false);
    assert!((3.29 - cost).abs() < 0.000000000001);

    // EQUAL lengths + sq_l2_dist
    let cost = naive::dtw(&data, &query_equal, dtw_cost::sq_l2_dist_vec, false);
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
        &dtw_cost::sq_l2_dist_vec,
    );
    assert!((4.5969 - cost_ucr).abs() < 0.000000000001);
}

#[test]
fn ucr_equals_naive_dtw() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    // Create random sequences for the query and the data time series
    // The observations are of type f64
    // The time series length is between 0 and 300
    let (data, query) = make_rdm_series((800, 900), None);
    // Creates dummy cummulative lower bound
    // We want the full calculation without abandoning or pruning so it consists only of 0.0
    let cb = vec![0.0; data.len()];
    let cost_ucr = ucr::dtw(&data, &query, &cb, data.len() - 2, f64::INFINITY, &cost_fn);
    let cost_naive = naive::dtw(&data, &query, cost_fn, false);
    assert!((cost_naive - cost_ucr).abs() < 0.000000000001);
}

#[test]
fn ucr_equals_improved_dtw() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    for _ in 0..100 {
        // Create random sequences for the query and the data time series
        // The observations are of type f64
        // The time series length is between 0 and 300
        let (data, query, cb_data, cb_query, w, bsf) = make_rdm_params((800, 900), None);

        let cost_ucr = ucr::dtw(&data, &query, &cb_query, w, bsf, &cost_fn);
        let cost_ucr_improved = ucr_improved::dtw(&data, &query, &cb_query, w, bsf, &cost_fn);

        println!("Testing with w: {}", w);
        println!("UCR     : {}", cost_ucr);
        println!("Improved: {}", cost_ucr_improved);
        if cost_ucr_improved.is_finite() && cost_ucr.is_finite() {
            assert!((cost_ucr_improved - cost_ucr).abs() < 0.000000000001);
        }
    }
}

#[test]
fn improved_dtw() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    // Create random sequences for the query and the data time series
    // The observations are of type f64
    // The time series length is between 0 and 300
    let data = [3., 1., 4., 4., 1., 1.];
    let query = [1., 3., 2., 1., 2., 2.];
    let w0 = 0;
    let w1 = 1;
    let w3 = 3;
    let cb_null = vec![0.0; data.len()];
    let bsf_six = 6.0;
    let bsf_nine = 9.10;
    let bsf_max = f64::MAX;

    // ###### Sakoe-Chiba band variation ################
    // w = 0
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, &cb_null, w0, bsf_nine, &cost_fn);
    assert!(cost_ucr_improved.is_infinite());

    // w = 1
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, &cb_null, w1, bsf_nine, &cost_fn);
    assert!(cost_ucr_improved.is_infinite());

    // w = query.len()
    let cost_ucr_improved =
        ucr_improved::dtw(&data, &query, &cb_null, query.len(), bsf_max, &cost_fn);
    assert!((cost_ucr_improved - 9.0).abs() < 0.000000000001);

    // w = query.len()+3
    let cost_ucr_improved =
        ucr_improved::dtw(&data, &query, &cb_null, query.len() + 3, bsf_max, &cost_fn);
    assert!((cost_ucr_improved - 9.0).abs() < 0.000000000001);

    // w = 3
    println!();
    println!("Improved Test 1");
    println!("UB : {:.2}", cb_null[0]);
    println!("w  : {:.2}", w3);
    println!("bsf: {:.2}", bsf_nine);
    println!("Matrix:");
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, &cb_null, w3, bsf_nine, &cost_fn);
    println!();
    println!("DTW dist: {}", cost_ucr_improved);
    assert!((cost_ucr_improved - 9.0).abs() < 0.000000000001);

    println!();
    println!("Improved Test 2");
    println!("UB = {:.2}", cb_null[0]);
    println!("w  : {:.2}", w3);
    println!("bsf: {:.2}", bsf_six);
    println!("Matrix:");
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, &cb_null, w3, bsf_six, &cost_fn);
    println!();
    println!("DTW dist: {}", cost_ucr_improved);
    assert!(cost_ucr_improved.is_infinite());
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

fn make_rdm_params(
    query_rng: (u32, u32),
    data_rng: Option<(u32, u32)>,
) -> (
    std::vec::Vec<f64>,
    std::vec::Vec<f64>,
    std::vec::Vec<f64>,
    std::vec::Vec<f64>,
    usize,
    f64,
) {
    use rand::thread_rng;
    use rand::Rng;

    let (query, data) = make_rdm_series(query_rng, data_rng);

    let (mut cb_query, mut cb_data) = (Vec::new(), Vec::new());

    let mut rng = thread_rng();
    let mut observation: f64;
    for i in 0..query.len() {
        observation = rng.gen_range(0.0..0.1);
        if i == 0 {
            cb_query.push(observation);
        } else {
            cb_query.push(observation + cb_query[i - 1]);
        }
    }
    for i in 0..data.len() {
        observation = rand::random();
        if i == 0 {
            cb_data.push(observation);
        } else {
            cb_data.push(observation + cb_data[i - 1]);
        }
    }
    let bsf = rng.gen_range(1000.0..2000.0);

    let w = rng.gen_range(0..query.len() - 2);

    (query, data, cb_query, cb_data, w, bsf)
}

fn make_test_series(equal_len: bool) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let a1 = vec![1.0, 1.];
    let a2 = vec![2.0, 1.];
    let a3 = vec![3.0, 1.];
    let a4 = vec![2.0, 1.];
    let a5 = vec![2.13, 1.];
    let a6 = vec![1.0, 1.];

    let b1 = vec![1.0, 1.];
    let b2 = vec![1.0, 1.];
    let b3 = vec![2.0, 1.];
    let b4 = vec![2.0, 1.];
    let b5 = vec![2.42, 1.];
    let b6 = vec![3.0, 1.];
    let b7 = vec![2.0, 1.];
    let b8 = vec![1.0, 1.];
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
