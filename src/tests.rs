#[cfg(test)]
use super::*;
#[test]
fn naive_dtw() {
    let (data, query) = utilities::make_test_series();
    let cost = naive::dtw(&data, &query, l2_dist, false);
    println!("naive squared cost: {}", cost);
    assert!((0.55 - cost).abs() < 0.000000000001);
}
#[test]
fn ucr_usp_dtw() {
    let (data, query) = utilities::make_test_series();
    // Creates dummy cummulative lower bound
    let cb = vec![f64::INFINITY; data.len()];
    let cost_ucr = ucr::dtw(&data, &query, &cb, data.len() - 2, f64::INFINITY, &l2_dist);
    let cost_naive = naive::dtw(&data, &query, l2_dist, false);
    println!("ucr cost: {}", cost_ucr);
    println!("naive cost: {}", cost_naive);
    assert!((0.55 - cost_ucr).abs() < 0.000000000001);
}
