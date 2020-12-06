# DTW
This is an implementation of the DTW algorithm in Rust. It is possible to calculate the similarity of two sequences of vectors of n components. You can provide a closure/function to calculate the cost.

## Usage
Add the dependency to your Cargo.toml and you can use the provided functions
```rust
use dtw::*;

fn main() {
    let (series_1, series_2) = utilities::make_test_series();
    let cost = dtw_naive(&series_1, &series_2, l2_dist, true);
    println!("cost: {}", cost);
}
```

## TODO
So far, there are no optimizations, no pruning... so it is not very efficient
