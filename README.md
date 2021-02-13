[![dependency status](https://deps.rs/repo/github/grelltrier/dtw/status.svg)](https://deps.rs/repo/github/grelltrier/dtw)
![Build](https://github.com/grelltrier/dtw/workflows/Build/badge.svg)

# DTW
This is an implementation of the DTW algorithm in Rust. It is possible to calculate the similarity of two sequences of observations. You can provide a closure/function to calculate the cost between observations. You could for example use the squared euclidean distance if your observations are points or n dimensions. The fastest implementation is in the module ucr_improved

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

## Potential for improvement
- For multi-dimensional data, when calculating the ED, it could abandon early by checking if the squared distance of one of the dimensions already exceeds
  the bsf of the best cell in that row
- Save space -> cost/cost_prev of len w instead to achieve a space complexity of O(4*w)
