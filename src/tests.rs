use super::*;

pub mod test_seq;

#[test]
// Test case A1
// Tests the naive DTW with different cost functions for sequences of EQUAL and UNEQUAL lengths
fn naive_dtw() {
    // ##### Sequences of UNEQUAL length #######
    let (query_unequal, data) = test_seq::make_test_series(false);

    // UNEQUAL lengths + l2_dist
    let cost = naive::dtw(&data, &query_unequal, dtw_cost::l2_dist_vec);
    assert!((0.55 - cost).abs() < 0.000000000001);

    // UNEQUAL lengths + sq_l2_dist
    let cost = naive::dtw(&data, &query_unequal, dtw_cost::sq_l2_dist_vec);
    assert!((0.19329999999999 - cost).abs() < 0.000000000001);

    // ##### Sequences of EQUAL length #######
    let (query_equal, data) = test_seq::make_test_series(true);
    // EQUAL lengths + l2_dist
    let cost = naive::dtw(&data, &query_equal, dtw_cost::l2_dist_vec);
    assert!((3.29 - cost).abs() < 0.000000000001);

    // EQUAL lengths + sq_l2_dist
    let cost = naive::dtw(&data, &query_equal, dtw_cost::sq_l2_dist_vec);
    assert!((4.5969 - cost).abs() < 0.000000000001);
}

#[test]
// Test case A2
fn ucr_usp_dtw() {
    let (query, data) = test_seq::make_test_series(true);
    // Creates dummy cummulative lower bound
    // We want the full calculation without abandoning or pruning so it consists only of 0.0
    let cost_ucr_usp = ucr_usp::dtw(
        &data,
        &query,
        None,
        data.len() - 2,
        f64::INFINITY,
        &dtw_cost::sq_l2_dist_vec,
    );
    assert!((4.5969 - cost_ucr_usp).abs() < 0.000000000001);
}

#[test]
// Test case A3
// Compares results of UCR-USP DTW and naive DTW for 100 random sequences
fn ucr_usp_equals_naive_dtw() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    for _ in 0..100 {
        // Create random sequences for the query and the data time series
        // The observations are of type f64
        // The time series length is between 0 and 300
        let (data, query) = test_seq::make_rdm_series((800, 900), None);
        // Creates dummy cummulative lower bound
        // We want the full calculation without abandoning or pruning so it consists only of 0.0
        let cost_ucr_usp =
            ucr_usp::dtw(&data, &query, None, data.len() - 2, f64::INFINITY, &cost_fn);
        let cost_naive = naive::dtw(&data, &query, cost_fn);
        assert!(
            cost_naive.is_infinite() && cost_ucr_usp.is_infinite()
                || (cost_naive - cost_ucr_usp).abs() < 0.000000000001
        );
    }
}

#[test]
// Test case A4
// Compares results of UCR-USP DTW and RPruned DTW for 100 random sequences
fn ucr_usp_equals_pruned_dtw() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    for _ in 0..100 {
        // Create random sequences for the query and the data time series
        // The observations are of type f64
        // The time series length is between 0 and 300
        let (data, query, _, cb_query, w, bsf) = test_seq::make_rdm_params((800, 900), None);

        let cost_ucr_usp = ucr_usp::dtw(&data, &query, Some(&cb_query), w, bsf, &cost_fn);
        let cost_rpruned = rpruned::dtw(&data, &query, Some(&cb_query), w, bsf, &cost_fn);

        println!("Testing with w: {}", w);
        println!("UCR_USP: {}", cost_ucr_usp);
        println!("RPruned: {}", cost_rpruned);
        assert!(
            cost_rpruned.is_infinite() && cost_ucr_usp.is_infinite()
                || (cost_rpruned - cost_ucr_usp).abs() < 0.000000000001
        );
    }
}

#[test]
// Test case A5
// Compares results of UCR-USP DTW and RPruned DTW for 100 random sequences.
// RPruned takes an iterator as input for the query sequence.
fn ucr_equals_improved_iter_dtw() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    for _ in 0..100 {
        // Create random sequences for the query and the data time series
        // The observations are of type f64
        // The time series length is between 0 and 300
        let (data, query, _, cb_query, w, bsf) = test_seq::make_rdm_params((800, 900), None);

        let cost_ucr_usp = ucr_usp::dtw(&data, &query, Some(&cb_query), w, bsf, &cost_fn);
        let cost_rpruned_iter =
            rpruned_iter::dtw(&data, query.into_iter(), Some(&cb_query), w, bsf, &cost_fn);

        println!("Testing with w: {}", w);
        println!("UCR_USP: {}", cost_ucr_usp);
        println!("RPruned: {}", cost_rpruned_iter);
        assert!(
            cost_rpruned_iter.is_infinite() && cost_ucr_usp.is_infinite()
                || (cost_rpruned_iter - cost_ucr_usp).abs() < 0.000000000001
        );
    }
}

#[test]
// Test case A6
// Compares results of UCR-USP DTW and RPruned DTW for a sequence where
// the first possible match in the last row is also the last cell
fn ucr_usp_equals_pruned_matching_in_very_last_cell_in_last_row() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let query = test_seq::make_knn_fail_query();
    let w = 12;

    // First problematic candidate sequence
    let bsf1 = 277.270;
    let data = test_seq::make_knn_fail_candidate(1);
    let cb1 = test_seq::make_knn_fail_cb1();
    let cost_ucr_usp = ucr_usp::dtw(&data, &query, Some(&cb1), w, bsf1, &cost_fn);
    let cost_rpruned = rpruned::dtw(&data, &query, Some(&cb1), w, bsf1, &cost_fn);

    println!("UCR_USP: {}", cost_ucr_usp);
    println!("RPruned: {}", cost_rpruned);
    assert!(
        cost_rpruned.is_infinite() && cost_ucr_usp.is_infinite()
            || (cost_rpruned - cost_ucr_usp).abs() < 0.000000000001
    );

    // Second problematic candidate sequence
    let bsf2 = 247.213;
    let data = test_seq::make_knn_fail_candidate(2);
    let cb2 = test_seq::make_knn_fail_cb2();
    let cost_ucr_usp = ucr_usp::dtw(&data, &query, Some(&cb2), w, bsf2, &cost_fn);
    let cost_rpruned = rpruned::dtw(&data, &query, Some(&cb2), w, bsf2, &cost_fn);

    println!("UCR_USP: {}", cost_ucr_usp);
    println!("RPruned: {}", cost_rpruned);
    assert!(
        cost_rpruned.is_infinite() && cost_ucr_usp.is_infinite()
            || (cost_rpruned - cost_ucr_usp).abs() < 0.000000000001
    );
}

#[test]
// Test case A7
// Compares results of UCR-USP DTW and RPruned DTW for a
// sequence where the last row is pruned
fn ucr_usp_equals_pruned_pruning_in_last_row() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let query = test_seq::make_knn_fail_query();
    let w = 12;

    let bsf = 277.270;
    let data = test_seq::make_knn_fail_candidate(3);
    let cb = test_seq::make_knn_fail_cb3();
    let cost_ucr_usp = ucr_usp::dtw(&data, &query, Some(&cb), w, bsf, &cost_fn);
    let cost_rpruned = rpruned::dtw(&data, &query, Some(&cb), w, bsf, &cost_fn);

    println!("UCR_USP: {}", cost_ucr_usp);
    println!("RPruned: {}", cost_rpruned);
    assert!(cost_rpruned.is_infinite() && cost_ucr_usp.is_infinite());
}

#[test]
// Test case A8
// Calculates RPruned DTW where the last value in cumulative
// bound is used for pruning
fn pruned_pruning_with_last_value_in_cb() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let query = test_seq::make_knn_fail_query();
    let w = 12;
    let bsf = 186.719;
    let data = test_seq::make_knn_fail_candidate(4);
    let cb = test_seq::make_knn_fail_cb4();
    let cost_rpruned = rpruned::dtw(&data, &query, Some(&cb), w, bsf, &cost_fn);
    println!("RPruned: {}", cost_rpruned);
    assert!(cost_rpruned.is_infinite());
}

#[test]
// Test case A9
// Calculates UCR-USP DTW where the last value in cumulative
// bound is used for pruning
fn ucr_usp_pruning_with_last_value_in_cb() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let query = test_seq::make_knn_fail_query();
    let w = 12;
    let bsf = 186.719;
    let data = test_seq::make_knn_fail_candidate(4);
    let cb = test_seq::make_knn_fail_cb4();
    let cost_ucr_usp = ucr_usp::dtw(&data, &query, Some(&cb), w, bsf, &cost_fn);
    println!("UCR_USP: {}", cost_ucr_usp);
    assert!(cost_ucr_usp.is_infinite());
}

#[test]
// Test case A10
// Calculates RPruned DTW for two simple sequences and vary the
// warping band and bsf
fn rpruned_warping_band() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let data = [3., 1., 4., 4., 1., 1.];
    let query = [1., 3., 2., 1., 2., 2.];
    let w0 = 0;
    let w1 = 1;
    let w3 = 3;
    let bsf_six = 6.0;
    let bsf_nine = 9.10;

    // ###### Sakoe-Chiba band variation ################
    // w = 0
    let cost_rpruned = rpruned::dtw(&data, &query, None, w0, bsf_nine, &cost_fn);
    assert!(cost_rpruned.is_infinite());

    // w = 1
    let cost_rpruned = rpruned::dtw(&data, &query, None, w1, bsf_nine, &cost_fn);
    assert!(cost_rpruned.is_infinite());

    // w = 3
    println!();
    println!("Pruned Test 1");
    println!("UB : {:.2}", 0.0);
    println!("w  : {:.2}", w3);
    println!("bsf: {:.2}", bsf_nine);
    println!("Matrix:");
    let cost_rpruned = rpruned::dtw(&data, &query, None, w3, bsf_nine, &cost_fn);
    println!();
    println!("DTW dist: {}", cost_rpruned);
    assert!((cost_rpruned - 9.0).abs() < 0.000000000001);

    println!();
    println!("Pruned Test 2");
    println!("UB = {:.2}", 0.0);
    println!("w  : {:.2}", w3);
    println!("bsf: {:.2}", bsf_six);
    println!("Matrix:");
    let cost_rpruned = rpruned::dtw(&data, &query, None, w3, bsf_six, &cost_fn);
    println!();
    println!("DTW dist: {}", cost_rpruned);
    assert!(cost_rpruned.is_infinite());

    // w = query.len()-2
    let cost_rpruned = rpruned::dtw(&data, &query, None, query.len() - 2, bsf_nine, &cost_fn);
    assert!((cost_rpruned - 9.0).abs() < 0.000000000001);
}

#[test]
#[should_panic]
// Test case A11
// Calculates RPruned DTW with a warping band equal to the query length
fn rpruned_w_equals_cb_len() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let data = [3., 1., 4., 4., 1., 1.];
    let query = [1., 3., 2., 1., 2., 2.];
    let cb = [1., 3., 2., 1., 2., 2.];
    let bsf_max = f64::MAX;

    // w = query.len()
    rpruned::dtw(&data, &query, Some(&cb), cb.len(), bsf_max, &cost_fn);
}

#[test]
#[should_panic]
// Test case A12
// Calculates RPruned DTW with a warping band greater than the query length
fn rpruned_w_greater_than_cb_len() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let data = [3., 1., 4., 4., 1., 1.];
    let query = [1., 3., 2., 1., 2., 2.];
    let cb = [1., 3., 2., 1., 2., 2.];
    let bsf_max = f64::MAX;

    // w = query.len() + 3
    rpruned::dtw(&data, &query, Some(&cb), cb.len() + 3, bsf_max, &cost_fn);
}

#[test]
// Test case A13
// Calculates RPruned, UCR-USP and naive DTW for sequences of unequal
// length with a warping band that is too small to yield any matches.
// If the warping window is too small, it becomes impossible for sequences
// of unequal length to have a distance other than infinity
fn warping_window_so_small_no_result_possible() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let data = [3., 1., 4., 4., 1., 1.];
    let query = [1., 3., 2., 1.];
    let bsf_max = f64::MAX;
    let w = 1;

    let cost_rpruned = rpruned::dtw(&query, &data, None, w, bsf_max, &cost_fn);
    assert!(cost_rpruned.is_infinite());

    let cost_ucr_usp = ucr_usp::dtw(&query, &data, None, w, bsf_max, &cost_fn);
    assert!(cost_ucr_usp.is_infinite());

    let cost_infinity = naive_with_w::dtw(&query, &data, w, &cost_fn);
    assert!(cost_infinity.is_infinite());
}
