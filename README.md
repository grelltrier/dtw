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
- Explore if it can be sped up for one-dimensional data by comparing the values within a row with the lowest found value before doing distance calculation
  Each row x serves to find the closest matching value of column y so we only need the distance of that one match
  To find it a comparison might be faster
  Could be implemented by the cost function. It could abandon early if its not smaller than the previous value
- For multi-dimensional data, when calculating the ED, it could abandon early by checking if the squared distance of one of the dimensions already exceeds
  the bsf of the best cell in that row