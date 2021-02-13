use super::*;

pub mod test_seq;

#[test]
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
fn ucr_usp_dtw() {
    let (query, data) = test_seq::make_test_series(true);
    // Creates dummy cummulative lower bound
    // We want the full calculation without abandoning or pruning so it consists only of 0.0
    let cost_ucr = ucr::dtw(
        &data,
        &query,
        None,
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
    let (data, query) = test_seq::make_rdm_series((800, 900), None);
    // Creates dummy cummulative lower bound
    // We want the full calculation without abandoning or pruning so it consists only of 0.0
    let cost_ucr = ucr::dtw(&data, &query, None, data.len() - 2, f64::INFINITY, &cost_fn);
    let cost_naive = naive::dtw(&data, &query, cost_fn);
    assert!(
        cost_naive.is_infinite() && cost_ucr.is_infinite()
            || (cost_naive - cost_ucr).abs() < 0.000000000001
    );
}

#[test]
fn ucr_equals_improved_dtw() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    for _ in 0..100 {
        // Create random sequences for the query and the data time series
        // The observations are of type f64
        // The time series length is between 0 and 300
        let (data, query, _, cb_query, w, bsf) = test_seq::make_rdm_params((800, 900), None);

        let cost_ucr = ucr::dtw(&data, &query, Some(&cb_query), w, bsf, &cost_fn);
        let cost_ucr_improved = ucr_improved::dtw(&data, &query, Some(&cb_query), w, bsf, &cost_fn);

        println!("Testing with w: {}", w);
        println!("UCR     : {}", cost_ucr);
        println!("Improved: {}", cost_ucr_improved);
        assert!(
            cost_ucr_improved.is_infinite() && cost_ucr.is_infinite()
                || (cost_ucr_improved - cost_ucr).abs() < 0.000000000001
        );
    }
}

#[test]
fn ucr_equals_improved_matching_in_very_last_cell_in_last_row() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let query = test_seq::make_knn_fail_query();
    let w = 12;

    // First problematic candidate sequence
    let bsf1 = 277.270;
    let data = test_seq::make_knn_fail_candidate(1);
    let cb1 = test_seq::make_knn_fail_cb1();
    let cost_ucr = ucr::dtw(&data, &query, Some(&cb1), w, bsf1, &cost_fn);
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, Some(&cb1), w, bsf1, &cost_fn);

    println!("UCR     : {}", cost_ucr);
    println!("Improved: {}", cost_ucr_improved);
    assert!(
        cost_ucr_improved.is_infinite() && cost_ucr.is_infinite()
            || (cost_ucr_improved - cost_ucr).abs() < 0.000000000001
    );

    // Second problematic candidate sequence
    let bsf2 = 247.213;
    let data = test_seq::make_knn_fail_candidate(2);
    let cb2 = test_seq::make_knn_fail_cb2();
    let cost_ucr = ucr::dtw(&data, &query, Some(&cb2), w, bsf2, &cost_fn);
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, Some(&cb2), w, bsf2, &cost_fn);

    println!("UCR     : {}", cost_ucr);
    println!("Improved: {}", cost_ucr_improved);
    assert!(
        cost_ucr_improved.is_infinite() && cost_ucr.is_infinite()
            || (cost_ucr_improved - cost_ucr).abs() < 0.000000000001
    );
}

#[test]
fn ucr_equals_improved_pruned_in_last_row() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let query = test_seq::make_knn_fail_query();
    let w = 12;

    let bsf = 277.270;
    let data = test_seq::make_knn_fail_candidate(3);
    let cb = test_seq::make_knn_fail_cb3();
    let cost_ucr = ucr::dtw(&data, &query, Some(&cb), w, bsf, &cost_fn);
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, Some(&cb), w, bsf, &cost_fn);

    println!("UCR     : {}", cost_ucr);
    println!("Improved: {}", cost_ucr_improved);
    assert!(
        cost_ucr_improved.is_infinite() && cost_ucr.is_infinite()
            || (cost_ucr_improved - cost_ucr).abs() < 0.000000000001
    );
}

#[test]
fn ucr_improved_pruned_with_last_value_in_cb() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let query = test_seq::make_knn_fail_query();
    let w = 12;
    let bsf = 186.719;
    let data = test_seq::make_knn_fail_candidate(4);
    let cb = test_seq::make_knn_fail_cb4();
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, Some(&cb), w, bsf, &cost_fn);
    println!("Improved: {}", cost_ucr_improved);
    assert!(cost_ucr_improved.is_infinite());
}

#[test]
fn ucr_pruned_with_last_value_in_cb() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let query = test_seq::make_knn_fail_query();
    let w = 12;
    let bsf = 186.719;
    let data = test_seq::make_knn_fail_candidate(4);
    let cb = test_seq::make_knn_fail_cb4();
    let cost_ucr = ucr::dtw(&data, &query, Some(&cb), w, bsf, &cost_fn);
    println!("Improved: {}", cost_ucr);
    assert!(cost_ucr.is_infinite());
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
    let bsf_six = 6.0;
    let bsf_nine = 9.10;

    // ###### Sakoe-Chiba band variation ################
    // w = 0
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, None, w0, bsf_nine, &cost_fn);
    assert!(cost_ucr_improved.is_infinite());

    // w = 1
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, None, w1, bsf_nine, &cost_fn);
    assert!(cost_ucr_improved.is_infinite());

    // w = 3
    println!();
    println!("Improved Test 1");
    println!("UB : {:.2}", 0.0);
    println!("w  : {:.2}", w3);
    println!("bsf: {:.2}", bsf_nine);
    println!("Matrix:");
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, None, w3, bsf_nine, &cost_fn);
    println!();
    println!("DTW dist: {}", cost_ucr_improved);
    assert!((cost_ucr_improved - 9.0).abs() < 0.000000000001);

    println!();
    println!("Improved Test 2");
    println!("UB = {:.2}", 0.0);
    println!("w  : {:.2}", w3);
    println!("bsf: {:.2}", bsf_six);
    println!("Matrix:");
    let cost_ucr_improved = ucr_improved::dtw(&data, &query, None, w3, bsf_six, &cost_fn);
    println!();
    println!("DTW dist: {}", cost_ucr_improved);
    assert!(cost_ucr_improved.is_infinite());

    // w = query.len()-2
    let cost_ucr_improved =
        ucr_improved::dtw(&data, &query, None, query.len() - 2, bsf_nine, &cost_fn);
    assert!((cost_ucr_improved - 9.0).abs() < 0.000000000001);
}

#[test]
#[should_panic]
fn ucr_improved_w_equals_shortest_len() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let data = [3., 1., 4., 4., 1., 1.];
    let query = [1., 3., 2., 1., 2., 2.];
    let bsf_max = f64::MAX;

    // w = query.len()
    ucr_improved::dtw(&data, &query, None, query.len(), bsf_max, &cost_fn);
}

#[test]
#[should_panic]
fn ucr_improved_w_greater_than_shortest_len() {
    let cost_fn = dtw_cost::sq_l2_dist_f64;
    let data = [3., 1., 4., 4., 1., 1.];
    let query = [1., 3., 2., 1., 2., 2.];
    let bsf_max = f64::MAX;

    // w = query.len() + 3
    ucr_improved::dtw(&data, &query, None, query.len() + 3, bsf_max, &cost_fn);
}
