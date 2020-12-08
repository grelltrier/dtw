use dtw::*;

fn main() {
    let (series_1, series_2) = utilities::make_test_series();
    let cost = naive::dtw(&series_1, &series_2, l2_dist, true);
    println!("cost: {}", cost);
}
