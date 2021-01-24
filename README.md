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
-  "c = cost_fn(&li[i], &co[j]);" should be inserted above line 36


- For multi-dimensional data, when calculating the ED, it could abandon early by checking if the squared distance of one of the dimensions already exceeds
  the bsf of the best cell in that row

- LB_Kim_FL only for the endpoints
  -> Change envelopes so they follow the boundary constraint and match first with first and last with last points
- Calculate the lower bound of the data sequence only when needed
- Use the z-normalized data sequence from the calculation of the lower bounds
- For one dimensional data, why bother squaring and not just taking absolute bounds as cost function?
- SS-PrunedDTW
  UB=bsf-cumLB[i+ws+1] <= shouldn't this just be UB=bsf-cumLB[i+1] instead???
  => The warping band is already taken into account when calculating the envelop (cumLB)
- lp in ssPrunedDTW can be set to n instead of i+1+ws?
- Save space -> cost/cost_prev of len w instead
- ec/next_ec never used for pruning??

- EAPrunedDTW
  "while j = next_start ∧ j < prev_pruningpoint do" should be "while j = next_start ∧ j <= prev_pruningpoint do"