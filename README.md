[![dependency status](https://deps.rs/repo/github/grelltrier/dtw/status.svg)](https://deps.rs/repo/github/grelltrier/dtw)
![Build](https://github.com/grelltrier/dtw/workflows/Build/badge.svg)

# DTW
This crate provides various implementations of the Dynamic Time Warping (DTW) algorithm in Rust. The algorithm can be used to calculate the similarity of two sequences of observations. You can provide a closure/function to calculate the cost between observations. You could for example use the squared euclidean distance if your observations are points or n dimensions. Some common cost functions can be found in the dtw_cost crate. The fastest implementation of DTW is in the module rpruned.

## Usage
Add the dependency to your Cargo.toml and you can use the provided functions
```rust
use dtw::*;

fn main() {
    let (query, data) = test_seq::make_test_series(true);
    let cost_rpruned = rpruned::dtw(
        &data,
        &query,
        None,
        data.len() - 2,
        f64::INFINITY,
        &dtw_cost::sq_l2_dist_vec,
    );
    println!("The DTW distance is {}", cost_rpruned);
}
```

## Potential for improvement
- For multi-dimensional data, when calculating the ED, it could abandon early by checking if the squared distance of one of the dimensions already exceeds
  the bsf of the best cell in that row
- Save space -> cost/cost_prev of len w instead to achieve a space complexity of O(4*w)